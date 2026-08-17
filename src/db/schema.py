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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from .types import DecimalText, JsonText
from sqlalchemy import (
    Column,
    CheckConstraint,
    ForeignKeyConstraint,
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
    Column("owner", Text, primary_key=True),
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

# The runtime's own plans, kept apart from `plan` on purpose — see
# `workspace/pilot_store.py` for why one row must not mean two things
# depending on which interpreter wrote it.
#
# Declared here because it exists in the database. It was created by a
# `CREATE TABLE IF NOT EXISTS` in the store and by no migration, so the first
# person to save a runtime plan added a table nothing had declared. The
# deployment then served happily until the next restart, at which point the
# schema-parity preflight — correctly — refused to start against a database it
# could not account for. A table that appears when a feature is first used is a
# landmine with a delay measured in whenever somebody deploys next.
pilot_plan = Table(
    "pilot_plans", metadata,
    # Owner first, like `plan_run`. Part of the identity, by the standing rule: a table with an owner keys
    # by it, or two tenants cannot hold the same id and an upsert lets one
    # overwrite the other. The table shipped without an owner because the
    # runtime pilot served one participant — a reason for the column to have
    # been absent, not a reason to key without it now that it exists.
    Column("owner", Text, primary_key=True),
    Column("plan_id", Text, primary_key=True),
    Column("created_at", Text, nullable=False),
    # The sentence, stored beside the intent rather than as the thing the plan
    # is made of.
    Column("text", Text, nullable=False),
    # The pinned artifact: intent, settled record, refusals by name.
    Column("artifact", Text, nullable=False),
)

# Clarification state, persisted so that answering a question is a write and
# looking at the result is a read.
#
# It exists because `POST /pilot/answer` rendered its result at the POST URL.
# A refresh, a Back, or a pasted link then issued a GET against a POST-only
# route and got `Method Not Allowed`, and Back returned to the last real GET —
# the empty form — losing everything the person had typed. The answers lived
# only in a request body.
#
# Separate from `pilot_plans` on purpose. A saved plan is something a person
# chose to keep; a review is where they are in the middle of being asked. They
# have different lifecycles, and merging them would put every half-answered
# attempt into "Your plans".
pilot_review = Table(
    "pilot_reviews", metadata,
    # Owner first, keyed with the id, by the standing rule for tenant-owned
    # tables: two participants must be able to hold the same content-addressed
    # review without one upserting over the other.
    Column("owner", Text, primary_key=True),
    Column("review_id", Text, primary_key=True),
    Column("created_at", Text, nullable=False),
    Column("text", Text, nullable=False),
    # The same pinned artifact `pilot_plans` stores: intent, settled record,
    # refusals. The GET path rebuilds from this and constructs no reader.
    Column("artifact", Text, nullable=False),
)

# The rest of the runtime's own tables, declared for the same reason. Each is
# created by its module on first use, so each appeared — or would appear — in a
# database partway through a deployment's life. `pilot_plans` is the one that
# actually fired; these three had not been used yet on this deployment, which
# is the only reason they had not.
# Two namespaces across all three, and keeping them apart is the point.
#
#   owner        the tenant. Decides who may read the row. In the key, as it is
#                on every other tenant-owned table, so one tenant's write can
#                never replace another's.
#   participant  the study pseudonym. Says which subject produced the row, and
#                nothing else. No foreign key joins it to an authenticated
#                user; a study that needs that mapping keeps it in its own
#                narrowly-held table.
#
# `participant` is in the key only on `pilot_consent`, because a consent record
# *is* the statement that one participant agreed — the pseudonym is intrinsic
# to what the row is. An event and a transcript merely happen to have been
# produced by one, so there the pseudonym stays out of the key and nullable.
#
# That is what lets the participant↔user association be deleted later without
# destroying or re-keying the experimental evidence. Putting the pseudonym into
# storage identity would have made deduplication, foreign keys, exports and
# restores all start depending on the one link that has to stay severable.
pilot_consent = Table(
    "pilot_consent", metadata,
    Column("owner", Text, primary_key=True),
    Column("participant", Text, primary_key=True),
    Column("state", Text, nullable=False),
    Column("at", Text, nullable=False),
    Column("notice_version", Text, nullable=False),
)

pilot_event = Table(
    "pilot_events", metadata,
    Column("owner", Text, primary_key=True),
    Column("event_id", Text, primary_key=True),
    Column("at", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("plan_id", Text),
    # Nullable, and added after the first version of this table — see
    # `pilot_events.ADDED_COLUMNS`, which repairs a database that has the
    # older shape.
    Column("participant", Text),
    Column("detail", Text, nullable=False),
)

pilot_transcript = Table(
    "pilot_transcripts", metadata,
    Column("owner", Text, primary_key=True),
    Column("entry_id", Text, primary_key=True),
    # Nullable so the pseudonym can be cleared from a transcript without
    # destroying the words. `every_participant` filters the NULLs out rather
    # than reporting a subject who never existed.
    Column("participant", Text),
    Column("at", Text, nullable=False),
    Column("attempt", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("detail", Text, nullable=False),
)

proposal = Table(
    "proposal", metadata,
    Column("proposal_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("owner", Text, primary_key=True),
    Column("payload", JsonText, nullable=False),
    Column("generated_at", Text, nullable=False),
    Column("status", Text, nullable=False),
)

observation = Table(
    "observation", metadata,
    Column("observation_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("owner", Text, primary_key=True),
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
    Column("owner", Text, primary_key=True),
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
    Column("owner", Text, primary_key=True),
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
    Column("owner", Text, primary_key=True),
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

# What a specific execution was actually delivered, as opposed to which source
# it was authorised to read. Append-only: a row here is a historical fact about
# a delivery that happened, and there is no state it can legitimately move to.
#
# `run_id` is a plain column, not a foreign key, and that is deliberate. The
# event is written *before* the run it names, so the run does not exist to point
# at yet — allocating the run identity first is what removes the second write
# that would otherwise be needed to connect them, and a chain with a second
# write has a half-connected state that a crash can find. The direction that
# does carry a constraint is `plan_run.access_event_id`, below.
market_data_access_event = Table(
    "market_data_access_event", metadata,
    Column("owner", Text, primary_key=True),
    Column("access_event_id", Text, primary_key=True),
    Column("request_id", Text, nullable=False),
    Column("run_id", Text),
    Column("snapshot_id", Text),
    # Which provenance record, not merely which snapshot: two provenances
    # differing only in access time are different records, and a snapshot id
    # cannot tell them apart.
    Column("provenance_digest", Text, nullable=False),
    Column("frame_digest", Text, nullable=False),
    Column("selected_columns", JsonText, nullable=False),
    Column("row_count", Integer, nullable=False),
    Column("range_start", Text),
    Column("range_end", Text),
    Column("policy_version", Text, nullable=False),
    Column("access_decision", Text, nullable=False),
    Column("accessed_at", Text, nullable=False),
    # What was asked for, beyond which snapshot. The snapshot says which
    # observations exist; this says which were delivered, and only the pair
    # determines the frame the digest describes. Nullable because events
    # written before it existed genuinely do not say — and a default here
    # would let them be "verified" against a frame they may never have carried.
    Column("resolution", JsonText),
    # Over the whole event body, so an edited field is detectable rather than
    # merely unlikely.
    Column("content_hash", Text, nullable=False),
)

Index("access_event_run", market_data_access_event.c.owner,
      market_data_access_event.c.run_id)

# The snapshot descriptor index. Shared reference data, and it carries no
# tenant column at all — see `Ownership.SHARED_REFERENCE`, whose rule is
# stricter than the tenant one rather than an exemption from it. A market
# snapshot describes the world; an `owner` here would mean nothing and would
# eventually be trusted by somebody.
#
# Keyed by `descriptor_hash`, not by `snapshot_hash`. The same observation
# bytes can legitimately have several descriptions over time — a licence
# re-reviewed, an adapter version corrected, provenance filled in — and each is
# a distinct record of what was believed about that data. The observations'
# identity stays fixed while the description moves, which is exactly the
# separation the contract draws.
market_snapshot = Table(
    "market_snapshot", metadata,
    Column("descriptor_hash", Text, primary_key=True),
    # Indexed rather than unique: several descriptors, one set of bytes.
    Column("snapshot_hash", Text, nullable=False),
    Column("snapshot_id", Text, nullable=False),
    Column("dataset_id", Text, nullable=False),
    Column("symbols", JsonText, nullable=False),
    Column("range_start", Text, nullable=False),
    Column("range_end", Text, nullable=False),
    Column("sessions", Integer, nullable=False),
    # What was asked for. Part of what produced the bytes, so a descriptor
    # without it would describe observations nobody could ask for again.
    Column("resolution", JsonText, nullable=False),
    Column("corporate_actions", Text, nullable=False),
    Column("calendar", Text, nullable=False),
    Column("source_adapter", Text, nullable=False),
    Column("source_adapter_version", Text, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("data_as_of", Text, nullable=False),
    Column("license_class", Text, nullable=False),
    Column("license_review_status", Text, nullable=False),
    Column("content_digest_version", Text, nullable=False),
    Column("contract_version", Text, nullable=False),
    Column("recorded_at", Text, nullable=False),
)

Index("market_snapshot_by_content", market_snapshot.c.snapshot_hash)

plan_migration = Table(
    "plan_migration", metadata,
    Column("owner", Text, primary_key=True),
    Column("migration_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("from_compiler", Text, nullable=False),
    Column("to_compiler", Text, nullable=False),
    Column("from_engine", Text, nullable=False),
    Column("to_engine", Text, nullable=False),
    Column("reason", Text, nullable=False),
    # Who agreed. Adopting a new interpretation changes what a saved plan
    # means, so the record names the person who accepted that rather than
    # implying the system decided.
    Column("authorized_by", Text, nullable=False),
    Column("migrated_at", Text, nullable=False),
    Column("old_run", Text),
    Column("new_run", Text),
    # The recompiled scenario. It lives here because `plan.scenario` is
    # immutable and keyed on the plan id — correctly, since the pinned parse is
    # the thing a user confirmed. So the superseding interpretation is stored
    # beside the authorisation for it rather than overwriting the original.
    Column("scenario", JsonText, nullable=False),
    Column("content_hash", Text, nullable=False),
)
"""One authorised recompilation of a saved plan.

The migration is provenance in its own right, not merely a step that produced
a run: it records what changed, why, who agreed, and which two runs sit either
side of it. Without it the new figure is simply newer than the old one.
"""

Index("plan_migration_plan", plan_migration.c.owner, plan_migration.c.plan_id)

run_invalidation = Table(
    "run_invalidation", metadata,
    Column("owner", Text, primary_key=True),
    Column("run_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    # The runtime's own vocabulary — RULE_NOT_EXECUTED — not prose. This is the
    # column an operator groups by when asking how far a defect reached, and
    # prose does not group.
    Column("classification", Text, nullable=False),
    Column("reason", Text, nullable=False),
    # Which engine produced the invalid run, so a replacement can say what
    # changed rather than merely being newer.
    Column("engine_version", Text, nullable=False),
    Column("invalidated_at", Text, nullable=False),
)
"""A stored run that must not be read as a strategy result.

The run itself is kept. Deleting it would destroy the evidence that the defect
happened and what was shown, and the correction here is not that a number was
mistyped — it is that the number answers a different question than the plan
asked. A user who remembers $5,160 must be able to find the record of having
been shown it.

**No foreign key to `plan_run`.** A retention policy that later purges runs
must not be blocked by, or silently delete, the record that one of them was
wrong. The pairing is by id and may outlive the row it names.
"""

Index("run_invalidation_plan", run_invalidation.c.owner,
      run_invalidation.c.plan_id)

plan_run = Table(
    "plan_run", metadata,
    Column("owner", Text, primary_key=True),
    Column("run_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("ran_at", Text, nullable=False),
    Column("result", JsonText, nullable=False),
    Column("comparison", JsonText, nullable=False),
    # Which delivery produced these figures. Nullable because runs recorded
    # before access events existed have none, and a placeholder would claim a
    # delivery nobody recorded — the same reason `NOT_RECORDED` is a status
    # rather than an empty provenance. A live producer may not leave it null;
    # that is enforced in `generate`, where the distinction between "historical
    # absence" and "this code declined to" is knowable.
    Column("access_event_id", Text),
    # `plan_run` used to carry no owner and was reachable only through its
    # plan. That made the production schema weaker than it needed to be —
    # tenant-unsafe run ids, a join on every ownership question, and an
    # ownership path that deletion, export and auditing each had to honour
    # separately. It is now directly scoped, and the composite foreign key
    # below makes a run belong to exactly one owner's plan by construction.
    #
    # `OwnershipPath.INDIRECT` keeps its own coverage in
    # `tests/ownership_fixture.py`, which owns an indirectly scoped table for
    # that purpose. Leaving a real domain table indirect to serve as a test
    # canary would have been paying for the test in production.
)


#: Which values each status column may hold, and the enum each one must match.
#:
#: Stated here rather than imported from those enums, for two reasons.
#:
#: The schema is imported by everything that touches storage, and the enums live
#: in modules that import storage — reading them here made a cycle. More
#: importantly, importing them would make the constraint agree with the enum *by
#: construction*, and a check that cannot disagree proves nothing. Declared
#: independently, `tests/test_referential_policy.py` compares the two and fails
#: when either moves without the other.
#:
#: These constrain *vocabulary*, not lifecycle. Whether PROPOSED may become
#: ACCEPTED, whether the required runs exist first, whether a proposal has gone
#: stale and whether a reconciliation may become MATCHED all stay in application
#: code, where the surrounding facts are available. A database constraint that
#: tried to express them would encode a partial version of a rule that lives
#: somewhere else, and the partial version would be the one enforced.
STATUS_VOCABULARY: Mapping[str, Tuple[str, ...]] = {
    "worksheet_proposal": ("PROPOSED", "ACCEPTED", "REJECTED", "EXPIRED",
                           "SUPERSEDED"),
    "proposal": ("OPEN", "ACCEPTED", "IGNORED", "EXPIRED", "SUPERSEDED"),
    "event_reconciliation": ("PENDING", "UNOBSERVED_OVERDUE", "MATCHED",
                             "MATCHED_WITH_VARIANCE", "LATE",
                             "MISSING_CONFIRMED", "UNEXPECTED", "AMBIGUOUS",
                             "CONFLICTING"),
    # Not an enum in the code: the store writes these two literals directly.
    # Listed here so the column is still constrained, and so the day someone
    # adds a third the constraint is what tells them to declare it.
    "worksheet_intent": ("PLANNED", "PROPOSED"),
}

#: Where each vocabulary's authoritative enum lives, for the test that compares
#: them. Kept as data so a new status column joins the comparison by being
#: declared rather than by someone remembering to extend a test.
STATUS_ENUMS: Mapping[str, Tuple[str, str]] = {
    "worksheet_proposal": ("src.workspace.apply", "ProposalStatus"),
    "proposal": ("src.mission.proposal", "ProposalStatus"),
    "event_reconciliation": ("src.mission.rsu_reconcile",
                             "ReconciliationStatus"),
}


class DeletePolicy(str, Enum):
    """What the database does when a referenced row is deleted.

    Stated per relationship rather than defaulted. A blanket `ON DELETE CASCADE`
    would make the database a second deletion model competing with
    `src/workspace/retention.py` — and the database's version would win silently,
    removing rows the application's own verification never saw and reporting
    success either way.
    """

    RESTRICT = "RESTRICT"
    """Refuse. The dependent row is independently meaningful — an audit record,
    a reconciliation, a run — and losing it with its parent would destroy
    history. This *reinforces* the application's deletion order instead of
    replacing it: delete the dependents first, as `delete_workspace` does, and
    the constraint is satisfied; forget one and the parent delete fails loudly
    rather than orphaning it."""

    CASCADE = "CASCADE"
    """Remove with the parent. Only where the dependent has no meaning at
    all without it."""

    SET_NULL = "SET NULL"
    """Keep the row, drop the link. For an optional reference whose historical
    record stays meaningful once the target is gone."""


@dataclass(frozen=True)
class Relationship:
    """One referential dependency, and the reason for its policy.

    Executable metadata, like `OwnershipPath`. The deletion order is derived
    from this graph rather than hand-sorted, so a new table joins the ordering
    by declaring its parent instead of by someone remembering to re-sort a list.
    """

    table: str
    columns: Tuple[str, ...]
    parent: str
    parent_columns: Tuple[str, ...]
    policy: DeletePolicy
    rationale: str

    def to_json(self) -> Dict[str, Any]:
        return {"table": self.table, "columns": list(self.columns),
                "parent": self.parent,
                "parent_columns": list(self.parent_columns),
                "policy": self.policy.value, "rationale": self.rationale}


#: Every referential dependency the schema enforces.
#:
#: Deliberately absent: `trace_id` on `worksheet_intent` and
#: `worksheet_proposal`. Traces are operational telemetry that expires on its
#: own schedule, and a foreign key would either block that expiry or delete
#: research records along with it. A trace that has aged out must leave a
#: readable intent behind — so the reference is allowed to dangle, and nothing
#: may require it to resolve.
#:
#: Also absent: `worksheet_id` on the worksheet-scoped tables. `worksheet` is
#: keyed on (owner, worksheet_id, revision) because revisions are immutable
#: history, so there is no single row a worksheet id refers to. Inventing a
#: unique key to hang a constraint on would be adding structure to satisfy a
#: constraint rather than to describe the domain. Those relationships stay
#: application-managed through `OwnershipPath`.
RELATIONSHIPS: Tuple[Relationship, ...] = (
    Relationship(
        table="plan_run", columns=("owner", "plan_id"),
        parent="plan", parent_columns=("owner", "plan_id"),
        policy=DeletePolicy.RESTRICT,
        rationale="A run is the record that a plan was executed and what it "
                  "produced. It outlives the interest in its plan, and "
                  "`retention.py` reaches it only through `plan.owner` — so the "
                  "constraint holds the application to deleting runs first, "
                  "which is the order `delete_workspace` already uses."),
    Relationship(
        table="plan_run", columns=("owner", "access_event_id"),
        parent="market_data_access_event",
        parent_columns=("owner", "access_event_id"),
        policy=DeletePolicy.RESTRICT,
        rationale="The event is the evidence that this run consumed this exact "
                  "frame. Cascading from it would let deleting the evidence "
                  "delete the finding, and RESTRICT is stronger still: while a "
                  "run exists, the delivery it cites cannot be removed, so a "
                  "stored figure can never become unverifiable by a deletion "
                  "elsewhere. The column is nullable — runs recorded before "
                  "access events existed cite none — and a foreign key does "
                  "not constrain NULL."),
    Relationship(
        table="event_reconciliation", columns=("owner", "worksheet_id",
                                               "planned_event_id"),
        parent="planned_event", parent_columns=("owner", "worksheet_id",
                                                "planned_event_id"),
        policy=DeletePolicy.RESTRICT,
        rationale="A reconciliation is the derived relationship between an "
                  "expectation and a report. Cascading from the expectation "
                  "would erase the finding that it was met or missed, which is "
                  "the only thing tracking is for."),
    Relationship(
        table="event_reconciliation", columns=("owner", "worksheet_id",
                                               "observed_event_id"),
        parent="observed_event", parent_columns=("owner", "worksheet_id",
                                                 "observed_event_id"),
        policy=DeletePolicy.RESTRICT,
        rationale="As above, from the observation side. The column is nullable "
                  "because pending, overdue and confirmed-missing rows have no "
                  "observation, and a foreign key does not constrain NULL."),
)


def _constraints() -> Tuple[ForeignKeyConstraint, ...]:
    """Build the constraints from the declared relationships.

    One source. A constraint written directly on a table would be a second
    declaration, and the two would disagree about a policy without anything
    noticing.
    """
    built = []
    for one in RELATIONSHIPS:
        built.append(ForeignKeyConstraint(
            [metadata.tables[one.table].c[name] for name in one.columns],
            [metadata.tables[one.parent].c[name] for name in one.parent_columns],
            ondelete=one.policy.value,
            name=f"fk_{one.table}_{'_'.join(one.columns)}"))
    return tuple(built)


for _constraint in _constraints():
    metadata.tables[_constraint.table.name].append_constraint(_constraint)


def _status_checks() -> None:
    """Constrain each status column to its declared vocabulary."""
    for table, values in STATUS_VOCABULARY.items():
        listed = ", ".join(f"'{value}'" for value in sorted(values))
        metadata.tables[table].append_constraint(
            CheckConstraint(f"status IN ({listed})",
                            name=f"ck_{table}_status"))


_status_checks()

#: Consistency that is genuinely local to one row, and nothing wider.
#:
#: A constraint spanning tables, or one that needed to know what else had
#: happened, would be a domain rule re-expressed incompletely in a place that
#: cannot see the rest of the facts.
metadata.tables["worksheet_proposal"].append_constraint(
    CheckConstraint(
        "result_revision IS NULL OR status = 'ACCEPTED'",
        name="ck_worksheet_proposal_result_only_when_accepted"))
# A rejected, expired or superseded proposal produced no revision. A row
# carrying one would claim an edit was applied that never was, and the
# worksheet history would not show it.

metadata.tables["event_reconciliation"].append_constraint(
    CheckConstraint(
        "observed_event_id IS NOT NULL OR status IN "
        "('PENDING', 'UNOBSERVED_OVERDUE', 'MISSING_CONFIRMED')",
        name="ck_event_reconciliation_observation_required"))
# Only three states are reached without an observation, and all three mean
# something specific about its absence. Any other status without an observed
# event is a conclusion drawn from nothing — the `unknown is not false`
# distinction, enforced at rest.


def deletion_order() -> Tuple[str, ...]:
    """Tables ordered so every dependent is deleted before its parent.

    Derived from `RELATIONSHIPS` rather than hand-sorted. The previous ordering
    was a heuristic — indirectly-owned tables first — which happened to be right
    for the one indirect table that existed and says nothing about a second.
    """
    remaining = dict.fromkeys(sorted(metadata.tables))
    parents: Dict[str, Set[str]] = {name: set() for name in remaining}
    for one in RELATIONSHIPS:
        if one.parent != one.table:
            parents[one.table].add(one.parent)

    ordered: List[str] = []
    while remaining:
        # A table is ready once nothing still waiting depends on it.
        ready = sorted(name for name in remaining
                       if not any(other != name and name in parents[other]
                                  for other in remaining))
        if not ready:
            raise RuntimeError(
                f"the relationship graph has a cycle among {sorted(remaining)}; "
                "no deletion order exists")
        for name in ready:
            ordered.append(name)
            del remaining[name]
    return tuple(ordered)


def primary_key_columns(table_name: str) -> Tuple[str, ...]:
    """The primary key of a table, from the model rather than from a guess.

    The upsert translation needs a conflict target, and a hand-maintained list
    of conflict targets is a second schema that drifts from the first.
    """
    table = metadata.tables[table_name]
    return tuple(column.name for column in table.primary_key.columns)


def table_names() -> Tuple[str, ...]:
    return tuple(sorted(metadata.tables))
