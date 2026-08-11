"""EvaluationProtocol — the second half of a published figure.

    methodology + evaluation protocol = performance

not ``methodology = performance``. A methodology says what to hold; a protocol
says how the holding was measured. Both determine the number, so both must be
versioned, hashed, and cited by the result.

Before this existed, the protocol was ambient execution state: transaction costs
and execution lag sat in ``config.py``, the walk-forward grid was implied by CLI
flags, and the data snapshot was whatever happened to be on disk. All of those
move a published return, and none of them appeared in the run record. A figure
was therefore reproducible only by accident.

Making the protocol a hashable artifact has a second effect that matters for
Release 2: **searching over protocols is multiple testing.** Trying ten
embargo settings and publishing the best is ten trials, exactly as trying ten
lookbacks is. Because the protocol is now an identified object, the trial counter
can see that search instead of attributing every run to the methodology alone.

Conventions match ``methodology.spec`` — frozen dataclasses, canonical JSON,
sha256 over semantic content, ``<kind>/<name>@<version>`` identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence

PROTOCOL_SPEC_VERSION = "0.1"


@dataclass(frozen=True)
class DataSnapshot:
    """Which data, as of when.

    ``content_hash`` is the digest of the actual price panel used. Vendors restate
    history — yfinance re-adjusts ``Adj Close`` for splits and dividends — so a
    date range alone does not identify the data. Two runs over "2016-01-01 to
    2025-11-20" can see different numbers.
    """

    source: str                       # e.g. "yfinance"
    start: str
    end: str
    content_hash: Optional[str] = None
    as_of: Optional[str] = None       # when the snapshot was taken
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "content_hash": self.content_hash,
            "as_of": self.as_of,
            "notes": self.notes,
        }

    def declared_form(self) -> Dict[str, Any]:
        """The declaration, without the realized panel hash.

        ``content_hash`` is a *run-time* fact: it is whatever the vendor served
        when the evaluation ran. Including it in protocol identity would mint a
        new protocol every time the vendor restated history, even though the
        procedure never changed. The realized hash is recorded on the run instead,
        so two runs of one protocol over restated data are correctly reported as
        the same procedure applied to different data — which is exactly how a
        restatement becomes visible.
        """
        return {
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "as_of": self.as_of,
        }


@dataclass(frozen=True)
class WalkForward:
    """The evaluation grid.

    ``purge`` and ``embargo`` are López de Prado's controls for label overlap and
    serial correlation. They default to zero only so that a protocol which does
    not need them can say so explicitly; a protocol evaluating an ML methodology
    with zero purge is making a claim, not accepting a default.
    """

    scheme: str = "expanding"         # expanding | rolling | none
    warmup: int = 252                 # trading days before the first evaluation
    step: int = 5                     # trading days between evaluations
    purge: int = 0                    # observations dropped around the test boundary
    embargo: int = 0                  # observations dropped after the test window
    train_window: Optional[int] = None  # rolling only; None = expanding
    calendar: str = "nyse@1"
    """Reference to a versioned trading calendar (`calendar/<name>@<version>`).

    Previously a bare enum (`business_days`), which meant "Monday to Friday,
    holidays silently included as flat days" — not any real exchange. Referencing
    an artifact makes the sessions a result was measured over citable, hashable
    and independently checkable.
    """

    periods_per_year: Optional[int] = None
    """Override for the calendar's own `periods_per_year`. Normally ``None``:
    the calendar knows how many sessions it has, and duplicating the number here
    invites the two to disagree."""

    def to_json(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "warmup": self.warmup,
            "step": self.step,
            "purge": self.purge,
            "embargo": self.embargo,
            "train_window": self.train_window,
            "calendar": self.calendar,
            "periods_per_year": self.periods_per_year,
        }


@dataclass(frozen=True)
class Holdout:
    """A sealed out-of-sample window.

    ``sealed`` is the operative field: while true, the evaluation runner must not
    read this period. Unsealing is a logged event, once per protocol — that is
    what converts an honour-system norm into an enforced control, and it is the
    strongest available answer to backtest overfitting.
    """

    start: Optional[str] = None
    end: Optional[str] = None
    sealed: bool = False
    unlock_event: Optional[str] = None   # id of the logged unlock, once opened

    @property
    def defined(self) -> bool:
        return self.start is not None and self.end is not None

    def to_json(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "sealed": self.sealed,
            "unlock_event": self.unlock_event,
        }


@dataclass(frozen=True)
class TransactionCosts:
    """How trading was charged.

    Previously ``config.TRANSACTION_COST_BPS`` — a module constant that shaped
    every published number while appearing in no run record.
    """

    model: str = "flat_bps"
    bps: float = 10.0
    execution_lag_days: int = 1
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "bps": self.bps,
            "execution_lag_days": self.execution_lag_days,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RiskModel:
    """The covariance and factor model used to *evaluate* results.

    Distinct from any covariance estimator inside a methodology. A methodology's
    estimator drives allocation and therefore belongs to the methodology — moving
    it here would mean a methodology no longer determines its own holdings, which
    would break the executable-data property. This one is the evaluation-side
    model: factor neutralization, attribution and risk decomposition.
    """

    covariance: str = "sample"        # sample | exponential | ledoit_wolf
    covariance_span: Optional[int] = None
    factor_model: Optional[str] = None   # e.g. "ff5", "market_only"; None = no neutralization
    factors: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {
            "covariance": self.covariance,
            "covariance_span": self.covariance_span,
            "factor_model": self.factor_model,
            "factors": list(self.factors),
        }


@dataclass(frozen=True)
class EvaluationProtocol:
    """A versioned, hashable description of how a methodology was measured."""

    name: str
    version: int
    title: str
    data_snapshot: DataSnapshot
    walk_forward: WalkForward = field(default_factory=WalkForward)
    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts)
    risk_model: RiskModel = field(default_factory=RiskModel)
    holdout: Holdout = field(default_factory=Holdout)
    benchmark: Optional[str] = None
    derived_from: Optional[str] = None
    change_rationale: str = ""
    spec_version: str = PROTOCOL_SPEC_VERSION

    # ---- identity ---------------------------------------------------------

    @property
    def concept_id(self) -> str:
        return f"protocol/{self.name}"

    @property
    def protocol_id(self) -> str:
        return f"protocol/{self.name}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        """Semantic content only — `title` and `change_rationale` are excluded so
        editing prose does not mint a new protocol version."""
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "data_snapshot": self.data_snapshot.declared_form(),
            "walk_forward": self.walk_forward.to_json(),
            "transaction_costs": self.transaction_costs.to_json(),
            "risk_model": self.risk_model.to_json(),
            "holdout": self.holdout.to_json(),
            "benchmark": self.benchmark,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_form(), sort_keys=True, separators=(",", ":"), default=str
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    # ---- revision ---------------------------------------------------------

    def revise(self, *, change_rationale: str, **changes: Any) -> "EvaluationProtocol":
        if not change_rationale.strip():
            raise ValueError(
                "change_rationale is required — an unexplained protocol change "
                "silently reprices every result measured under it"
            )
        return replace(
            self,
            version=self.version + 1,
            derived_from=self.protocol_id,
            change_rationale=change_rationale,
            **changes,
        )

    def with_snapshot_hash(self, content_hash: str) -> "EvaluationProtocol":
        """Record which panel this evaluation actually saw.

        Called by the runner once the data is loaded. Deliberately does **not**
        change `content_hash` — see `DataSnapshot.declared_form`. The realized
        hash rides along for reporting and is persisted on the run record.
        """
        return replace(
            self, data_snapshot=replace(self.data_snapshot, content_hash=content_hash)
        )

    def unseal(self, unlock_event: str) -> "EvaluationProtocol":
        """Open the holdout. Once, and it is logged."""
        if not self.holdout.defined:
            raise ValueError("no holdout is defined on this protocol")
        if not self.holdout.sealed:
            raise ValueError(
                f"holdout already opened by {self.holdout.unlock_event!r} — "
                "a sealed holdout may be opened once"
            )
        return replace(
            self,
            holdout=replace(self.holdout, sealed=False, unlock_event=unlock_event),
        )

    def to_json(self) -> Dict[str, Any]:
        payload = self.canonical_form()
        payload.update(
            {
                "concept_id": self.concept_id,
                "protocol_id": self.protocol_id,
                "content_hash": self.content_hash,
                "title": self.title,
                "derived_from": self.derived_from,
                "change_rationale": self.change_rationale,
            }
        )
        return payload


def from_dict(payload: Mapping[str, Any]) -> EvaluationProtocol:
    """Parse a protocol from its JSON/YAML form."""
    snap = payload.get("data_snapshot", {})
    wf = payload.get("walk_forward", {})
    tc = payload.get("transaction_costs", {})
    rm = payload.get("risk_model", {})
    ho = payload.get("holdout", {})

    return EvaluationProtocol(
        name=payload["name"],
        version=int(payload["version"]),
        title=payload.get("title", payload["name"]),
        data_snapshot=DataSnapshot(
            source=snap.get("source", "unknown"),
            start=snap.get("start", ""),
            end=snap.get("end", ""),
            content_hash=snap.get("content_hash"),
            as_of=snap.get("as_of"),
            notes=snap.get("notes", ""),
        ),
        walk_forward=WalkForward(
            scheme=wf.get("scheme", "expanding"),
            warmup=int(wf.get("warmup", 252)),
            step=int(wf.get("step", 5)),
            purge=int(wf.get("purge", 0)),
            embargo=int(wf.get("embargo", 0)),
            train_window=wf.get("train_window"),
            calendar=wf.get("calendar", "nyse@1"),
            periods_per_year=(
                int(wf["periods_per_year"]) if wf.get("periods_per_year") else None
            ),
        ),
        transaction_costs=TransactionCosts(
            model=tc.get("model", "flat_bps"),
            bps=float(tc.get("bps", 10.0)),
            execution_lag_days=int(tc.get("execution_lag_days", 1)),
            notes=tc.get("notes", ""),
        ),
        risk_model=RiskModel(
            covariance=rm.get("covariance", "sample"),
            covariance_span=rm.get("covariance_span"),
            factor_model=rm.get("factor_model"),
            factors=tuple(rm.get("factors", ())),
        ),
        holdout=Holdout(
            start=ho.get("start"),
            end=ho.get("end"),
            sealed=bool(ho.get("sealed", False)),
            unlock_event=ho.get("unlock_event"),
        ),
        benchmark=payload.get("benchmark"),
        derived_from=payload.get("derived_from"),
        change_rationale=payload.get("change_rationale", ""),
        spec_version=payload.get("spec_version", PROTOCOL_SPEC_VERSION),
    )
