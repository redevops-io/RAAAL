"""The versioned ``SavedStrategyPlan`` — the handoff contract to Wealth Manager.

This is the object that crosses the boundary the moment a person presses **Save**
and later **Connect account**. It carries only *portable strategy meaning and its
provenance* — never broker authority. Wealth Manager imports it as evidence and
intent, and creates its own separate ``PolicyBinding`` / account authorization; it
must be impossible for this object to smuggle execution authority across.

It is built ON the existing export surface, not beside it:

  * the shared-form strategy meaning is ``runtime_export.payload_from_intent`` —
    extended here with contribution semantics — so there is one extraction, not two;
  * ``source_intent_hash`` is RAAAL's native ``intent.intent_hash``, carried
    **verbatim** (never recomputed), exactly as ``runtime_artifact_for`` does;
  * ``content_hash`` is the runtime-contracts ``rcv1`` ``content_hash``, the same
    canonical hash the boundary adapter uses — so a wealth-manager import that
    recomputes it agrees byte-for-byte with no shared code, only a shared contract;
  * ``to_runtime_artifact`` bridges to the existing dual-identity wire artifact, so
    a saved plan still composes with ``from_raaal.from_runtime_artifact``.

**Identity is meaning + the pinned methodology/protocol/data, nothing else.** The
``content_hash`` is computed over ``{schema_version, verified_strategy_intent,
strategy_constraints, methodology_id, methodology_version, protocol_version,
market_data_snapshot_id}``. It deliberately EXCLUDES the save envelope (plan_id,
owner, tenant), the timestamps, the evaluation refs, the amendment log, the
disclosure acknowledgement, and even ``source_intent_hash``:

  * the same evaluated strategy saved by two people, or saved twice, is the *same*
    strategy — the envelope is not part of what the strategy *is* (mirrors the
    boundary's "source excluded from identity" rule);
  * a changed methodology, protocol, or market-data snapshot is a *new* identity,
    because the numbers the user saw would differ (mirrors the golden-test rule
    "changed methodology/data snapshot → new identity").

Timestamps are inputs, never wall-clock read inside the contract, so the object is
deterministic and a fixture reproduces bit-for-bit on any machine.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import runtime_contracts as rc

from .runtime_export import payload_from_intent

#: The versioned contract name and version. Start at "v1"; a wire form carrying a
#: different version is migrated (older) or refused (unknown/newer), never guessed.
SCHEMA = "raaal/saved-strategy-plan"
SCHEMA_VERSION = "v1"

#: The intent fields that carry contribution semantics — an amount and its cadence.
#: Read here (not in ``payload_from_intent``) so the shared export seam keeps its
#: current shape while the saved plan carries the richer meaning it needs.
_AMOUNT = "amount"
_CONTRIBUTION_CADENCE = "cadence"
_STATED_WEIGHTS = "stated_weights"
_ASSETS = "assets"


class ForbiddenAuthorityError(ValueError):
    """A saved plan was carrying broker authority it must never contain.

    Raised the instant such a key is seen — on construction, on ``from_intent``,
    and on ``from_dict`` — so a caller cannot smuggle execution authority,
    credentials, tax lots, household restrictions, or account-specific suitability
    into the object that crosses to Wealth Manager. This object is meaning and
    provenance; authority is created on the far side, separately.
    """


class SchemaVersionError(ValueError):
    """A wire form declares a schema version this build cannot interpret.

    Refused rather than guessed: a newer or unknown version may mean something
    this code does not know, and a saved financial plan is the last place to
    silently assume it means what the old shape meant.
    """


#: Keys that would carry broker authority into the plan. Matched exactly and
#: case-insensitively against every dict key anywhere in the meaning/constraints,
#: so the guard is predictable (no false positive on a legitimate strategy key)
#: while still refusing the documented forbidden set (§9 "must not contain").
_FORBIDDEN_KEYS = frozenset({
    # brokerage credentials
    "brokerage_credentials", "broker_credentials", "credentials",
    "api_key", "api_secret", "access_token", "account_number",
    "account_credentials", "secret",
    # execution authorization
    "execution_authorization", "execution_authority", "trade_authorization",
    "authorization", "authorized_to_execute", "execution_grant",
    # inferred household restrictions
    "household_restrictions", "household", "inferred_restrictions",
    "household_constraints",
    # tax lots
    "tax_lots", "tax_lot", "lots", "cost_basis", "holdings",
    # account-specific suitability
    "suitability", "suitability_claim", "account_suitability",
    "suitability_profile",
})


def _assert_no_forbidden_authority(obj: Any, *, path: str = "") -> None:
    """Walk a nested structure and refuse any key that carries broker authority."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str) and key.strip().lower() in _FORBIDDEN_KEYS:
                where = f"{path}.{key}" if path else key
                raise ForbiddenAuthorityError(
                    f"a SavedStrategyPlan must not carry broker authority; the key "
                    f"{where!r} is forbidden — credentials, execution authorization, "
                    f"tax lots, inferred household restrictions and account-specific "
                    f"suitability are created by Wealth Manager, never carried here")
            _assert_no_forbidden_authority(value, path=f"{path}.{key}" if path else str(key))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _assert_no_forbidden_authority(item, path=f"{path}[{i}]")


def _named_weights(intent: Any) -> dict[str, float]:
    """Parse ``stated_weights`` in the ``NAME=weight`` form the catalog emits
    (e.g. ``"VTI=60,BND=40"`` → ``{"VTI": 0.6, "BND": 0.4}``), normalised to
    fractions. Returns ``{}`` when the strategy states no explicit named weights —
    the ``allocation_method`` then carries the meaning and the consumer derives the
    weights. Complements ``payload_from_intent``'s ``"60/40"`` + ``"a,b"`` parser."""
    field = getattr(intent, "fields", {}).get(_STATED_WEIGHTS)
    raw = "" if field is None else str(getattr(field, "value", field) or "")
    pairs: dict[str, float] = {}
    for token in raw.split(","):
        if "=" not in token:
            return {}
        name, _, weight = token.partition("=")
        name = name.strip()
        try:
            pairs[name] = float(weight.strip())
        except ValueError:
            return {}
        if not name:
            return {}
    total = sum(pairs.values())
    if not pairs or total <= 0:
        return {}
    return {name: weight / total for name, weight in pairs.items()}


def _contribution(intent: Any) -> dict[str, str]:
    """Contribution semantics — the amount and the cadence it recurs on — read
    from the native intent's own fields. Empty strings where unstated; never
    fabricated."""
    fields = getattr(intent, "fields", {})

    def value(name: str) -> str:
        field = fields.get(name)
        if field is None:
            return ""
        v = getattr(field, "value", field)
        return "" if v is None else str(v)

    return {"amount": value(_AMOUNT), "cadence": value(_CONTRIBUTION_CADENCE)}


def verified_strategy_intent_from(intent: Any, *, label: str = "") -> dict:
    """The confirmed strategy meaning: ``payload_from_intent``'s shared form
    (allocation_method, target_allocation, objective, rebalancing, universe, label)
    EXTENDED with contribution semantics. Reuses the existing extraction so RAAAL
    has one definition of what a strategy selection means, not two."""
    meaning = dict(payload_from_intent(intent, label=label))
    if not meaning.get("target_allocation"):
        # the catalog states weights as NAME=weight; recover them when the base
        # extractor (which reads the "60/40" form) found none.
        named = _named_weights(intent)
        if named:
            meaning["target_allocation"] = named
    meaning["contribution"] = _contribution(intent)
    return meaning


def _canonical_meaning(*, schema_version: str, verified_strategy_intent: Mapping,
                       strategy_constraints: Mapping, provenance: Mapping) -> dict:
    """Exactly the fields identity is computed over — meaning plus the pinned
    methodology/protocol/data. The save envelope, timestamps, evaluation refs,
    amendments, disclosure ack and ``source_intent_hash`` are all excluded."""
    return {
        "schema_version": schema_version,
        "verified_strategy_intent": dict(verified_strategy_intent),
        "strategy_constraints": dict(strategy_constraints),
        "methodology_id": str(provenance.get("methodology_id", "")),
        "methodology_version": str(provenance.get("methodology_version", "")),
        "protocol_version": str(provenance.get("protocol_version", "")),
        "market_data_snapshot_id": str(provenance.get("market_data_snapshot_id", "")),
    }


@dataclasses.dataclass(frozen=True)
class SavedStrategyPlan:
    """A saved strategy plan: portable meaning + provenance, never broker authority.

    Frozen and versioned. ``content_hash`` is computed from the canonical meaning at
    construction (via runtime-contracts ``rcv1``) and is not accepted from a caller —
    a wire form that disagrees with what its own meaning hashes to is rejected as
    tampered.
    """

    verified_strategy_intent: Mapping[str, Any]
    strategy_constraints: Mapping[str, Any]
    provenance: Mapping[str, Any]
    created_at: str
    effective_at: str
    schema: str = SCHEMA
    schema_version: str = SCHEMA_VERSION
    plan_id: str = ""
    plan_version: int = 1
    owner_id: str = ""
    tenant_id: str = ""
    amendments: Sequence[Mapping[str, Any]] = ()
    disclosure_acknowledgement: Optional[Mapping[str, Any]] = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise SchemaVersionError(
                f"not a {SCHEMA!r} contract: schema={self.schema!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported schema_version {self.schema_version!r}; this build "
                f"speaks {SCHEMA_VERSION!r}. Route older forms through migrate().")
        # No broker authority anywhere in the meaning, constraints or amendments.
        _assert_no_forbidden_authority(self.verified_strategy_intent,
                                       path="verified_strategy_intent")
        _assert_no_forbidden_authority(self.strategy_constraints,
                                       path="strategy_constraints")
        for i, amendment in enumerate(self.amendments):
            _assert_no_forbidden_authority(amendment, path=f"amendments[{i}]")

        computed = rc.content_hash(_canonical_meaning(
            schema_version=self.schema_version,
            verified_strategy_intent=self.verified_strategy_intent,
            strategy_constraints=self.strategy_constraints,
            provenance=self.provenance))
        if self.content_hash and self.content_hash != computed:
            raise ValueError(
                f"content_hash {self.content_hash!r} does not match the plan's own "
                f"canonical meaning ({computed!r}); the wire form is tampered or was "
                f"produced under a different contract")
        object.__setattr__(self, "content_hash", computed)
        # Normalise the mutable containers to immutable snapshots.
        object.__setattr__(self, "verified_strategy_intent",
                           dict(self.verified_strategy_intent))
        object.__setattr__(self, "strategy_constraints",
                           dict(self.strategy_constraints))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "amendments",
                           tuple(dict(a) for a in self.amendments))

    # -- construction -------------------------------------------------------

    @classmethod
    def from_intent(cls, intent: Any, *, label: str = "",
                    methodology: Mapping[str, Any],
                    protocol_version: str,
                    market_data_snapshot_id: str,
                    evaluation_result_refs: Sequence[Mapping[str, Any]] = (),
                    strategy_constraints: Optional[Mapping[str, Any]] = None,
                    created_at: str, effective_at: str,
                    owner_id: str = "", tenant_id: str = "",
                    plan_id: str = "", plan_version: int = 1,
                    disclosure_acknowledgement: Optional[Mapping[str, Any]] = None
                    ) -> "SavedStrategyPlan":
        """Build a plan from a **sealed** native intent.

        Refuses an unsealed intent, exactly as ``runtime_artifact_for`` does: with no
        ``intent_hash`` there is no native identity to carry as provenance, and an
        unsealed intent is not something the far side should build a policy on. The
        native hash is carried **verbatim** into ``provenance.source_intent_hash`` —
        never recomputed.
        """
        source = getattr(intent, "intent_hash", None)
        if not source:
            raise ValueError(
                "cannot save an unsealed intent as a strategy plan — it has no "
                "intent_hash to carry verbatim as source_intent_hash provenance")

        methodology = methodology or {}
        provenance = {
            "source_intent_hash": source,               # NATIVE, verbatim
            "methodology_id": str(methodology.get("id",
                                  methodology.get("methodology_id", ""))),
            "methodology_version": str(methodology.get("version",
                                       methodology.get("methodology_version", ""))),
            "protocol_version": str(protocol_version),
            "market_data_snapshot_id": str(market_data_snapshot_id),
            "evaluation_result_refs": [dict(r) for r in evaluation_result_refs],
        }
        return cls(
            verified_strategy_intent=verified_strategy_intent_from(intent, label=label),
            strategy_constraints=dict(strategy_constraints or {}),
            provenance=provenance,
            created_at=created_at, effective_at=effective_at,
            owner_id=owner_id, tenant_id=tenant_id,
            plan_id=plan_id, plan_version=plan_version,
            disclosure_acknowledgement=(dict(disclosure_acknowledgement)
                                        if disclosure_acknowledgement else None))

    # -- wire form ----------------------------------------------------------

    def to_dict(self) -> dict:
        """The versioned wire form, JSON-serialisable, in a fixed key order."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "verified_strategy_intent": dict(self.verified_strategy_intent),
            "strategy_constraints": dict(self.strategy_constraints),
            "provenance": dict(self.provenance),
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "effective_at": self.effective_at,
            "amendments": [dict(a) for a in self.amendments],
            "disclosure_acknowledgement": (dict(self.disclosure_acknowledgement)
                                           if self.disclosure_acknowledgement else None),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SavedStrategyPlan":
        """Rebuild a plan from its wire form, migrating an older version if needed
        and refusing an unknown/newer one. The stored ``content_hash`` is verified
        against the plan's own meaning (tamper detection) in ``__post_init__``."""
        data = migrate(dict(data))
        # Guard the incoming wire form as a whole before trusting any field.
        _assert_no_forbidden_authority(data)
        return cls(
            schema=data.get("schema", SCHEMA),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            plan_id=data.get("plan_id", ""),
            plan_version=int(data.get("plan_version", 1)),
            owner_id=data.get("owner_id", ""),
            tenant_id=data.get("tenant_id", ""),
            verified_strategy_intent=dict(data.get("verified_strategy_intent", {})),
            strategy_constraints=dict(data.get("strategy_constraints", {})),
            provenance=dict(data.get("provenance", {})),
            content_hash=data.get("content_hash", ""),
            created_at=data.get("created_at", ""),
            effective_at=data.get("effective_at", ""),
            amendments=tuple(dict(a) for a in data.get("amendments", ())),
            disclosure_acknowledgement=(dict(data["disclosure_acknowledgement"])
                                        if data.get("disclosure_acknowledgement")
                                        else None))

    # -- bridges & evolution ------------------------------------------------

    def to_runtime_artifact(self, *, produced_at: str = "") -> dict:
        """Bridge to the existing dual-identity runtime artifact.

        Reuses ``src.runtime_boundary.to_runtime_artifact`` so a saved plan still
        composes with the current boundary and with ``from_raaal.from_runtime_artifact``
        on the wealth-manager side: the native ``source_intent_hash`` is carried
        verbatim and the ``runtime_artifact_hash`` is the ``rcv1`` hash of the
        shared-form strategy selection.
        """
        from ..runtime_boundary import to_runtime_artifact as _to_artifact

        source = self.provenance.get("source_intent_hash", "")
        return _to_artifact(source_intent_hash=source,
                            payload=dict(self.verified_strategy_intent),
                            produced_at=produced_at)

    def amend(self, *, note: str, at: str,
              verified_strategy_intent: Optional[Mapping[str, Any]] = None,
              strategy_constraints: Optional[Mapping[str, Any]] = None,
              effective_at: Optional[str] = None,
              disclosure_acknowledgement: Optional[Mapping[str, Any]] = None
              ) -> "SavedStrategyPlan":
        """A new plan with ``plan_version`` bumped, this amendment appended, and the
        ``content_hash`` recomputed. The amendment records the *prior* content hash,
        so the chain of what-changed is auditable. Passing new meaning or constraints
        is what can change the identity; a note-only amendment keeps it."""
        new_version = self.plan_version + 1
        record = {
            "at": at,
            "plan_version": new_version,
            "note": note,
            "prior_content_hash": self.content_hash,
        }
        return dataclasses.replace(
            self,
            verified_strategy_intent=dict(verified_strategy_intent
                                          if verified_strategy_intent is not None
                                          else self.verified_strategy_intent),
            strategy_constraints=dict(strategy_constraints
                                      if strategy_constraints is not None
                                      else self.strategy_constraints),
            effective_at=effective_at if effective_at is not None else self.effective_at,
            plan_version=new_version,
            amendments=tuple(self.amendments) + (record,),
            disclosure_acknowledgement=(dict(disclosure_acknowledgement)
                                        if disclosure_acknowledgement is not None
                                        else self.disclosure_acknowledgement),
            content_hash="")   # force recompute in __post_init__


# --- versioning / migration -------------------------------------------------
#
# The same idiom as `migrate_plan.py`: an explicit table of upgrades keyed by the
# version being left, applied in order until the current version is reached. There
# is only one version today, so the table is empty — but the *shape* is here so a
# v2 is a row added deliberately, and an unknown/newer version is a refusal rather
# than a silent reinterpretation of a saved financial plan.

#: from_version -> function(dict) -> dict (raising the version by exactly one step).
_MIGRATIONS: dict = {}

#: Versions this build can reach, newest last. Extend as `_MIGRATIONS` grows.
_KNOWN_VERSIONS = ("v1",)


def migrate(data: Mapping[str, Any]) -> dict:
    """Upgrade an older wire form to the current ``SCHEMA_VERSION``; refuse an
    unknown or newer one. Never guesses what a version it does not know means."""
    data = dict(data)
    version = data.get("schema_version", SCHEMA_VERSION)
    if version == SCHEMA_VERSION:
        return data
    if version not in _KNOWN_VERSIONS:
        raise SchemaVersionError(
            f"unknown SavedStrategyPlan schema_version {version!r}; this build "
            f"knows {_KNOWN_VERSIONS!r} and refuses to guess a form it does not "
            f"recognise")
    seen = set()
    while data.get("schema_version") != SCHEMA_VERSION:
        current = data.get("schema_version")
        if current in seen or current not in _MIGRATIONS:
            raise SchemaVersionError(
                f"no migration path from schema_version {current!r} to "
                f"{SCHEMA_VERSION!r}")
        seen.add(current)
        data = _MIGRATIONS[current](data)
    return data
