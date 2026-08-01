"""Server-rendered pages that make the execution model legible.

Deliberately not a dashboard. The differentiator is not better charts — it is
that every number is attached to the artifacts that produced it, and that the
platform will say plainly when two figures should not be compared.

Success criterion: a reader should be able to work out *why* a methodology is
trustworthy, or why it is not, by navigating the pages — without reading the
documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..calendars import CalendarRegistry
from ..evaluation import ProtocolRegistry
from ..evaluation.runner import assess_compatibility, evaluate, periods_per_year
from ..knowledge import (
    AssumptionRegistry,
    ClaimRegistry,
    EvidenceRegistry,
    FindingRegistry,
    InvestigationRegistry,
    KnowledgeGraph,
    assess_claim,
)
from ..ledger import Ledger
from ..methodology import MethodologyRegistry
from ..policy import PolicyRegistry, Surface, decide
from ..statistics.assessment import assess
from ..statistics.neutralize import FactorModel
from .chain import CHAIN_ORDER, build_chain_state, chain_from_record
from .comparability import build_comparability_view
from .drift import evaluate_drift
from .semantics import resolve as resolve_relation_semantics
from .graph import (
    assumption_dependency_projection,
    version_timeline,
    claim_stance_projection,
    finding_impact_projection,
)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

#: Templates may *look up* a relation's semantics; they may never *decide* them.
#: Exposing the registry itself is what makes that distinction enforceable — a
#: template that hardcodes "INVALIDATES_RESULTS_OF means red" has authored an
#: opinion, while one calling `relation(...)` is reading the single declaration.
#: The lookup raises on an undeclared relation, so a new relation type cannot
#: reach a page as a silent default.
TEMPLATES.env.globals["relation"] = resolve_relation_semantics
router = APIRouter(prefix="/ui", tags=["ui"])

PRICES = Path("data/history/prices.parquet")
DEFAULT_PROTOCOL = "standard@1"
DEFAULT_POLICY = "library-default@1"


def _registries():
    return (
        MethodologyRegistry(),
        ProtocolRegistry(),
        PolicyRegistry(),
        CalendarRegistry(),
        Ledger(),
    )


def _prices() -> Optional[pd.DataFrame]:
    return pd.read_parquet(PRICES) if PRICES.exists() else None


def _pick_protocol(methodology, protocols: ProtocolRegistry):
    """Choose a protocol that can validly evaluate this methodology.

    A page must not silently render a result from an incompatible pairing — that
    is precisely the mislabelling `assess_compatibility` exists to prevent.
    """
    for protocol in protocols.load_all():
        if assess_compatibility(methodology, protocol).compatible:
            return protocol
    return None


def _graph() -> KnowledgeGraph:
    """Build the knowledge graph from the on-disk registries."""
    return KnowledgeGraph(
        MethodologyRegistry().load_all(),
        ClaimRegistry().load_all(),
        AssumptionRegistry().load_all(),
        EvidenceRegistry().load_all(),
        FindingRegistry().load_all(),
        InvestigationRegistry().load_all(),
    )


_LINEAGE_CACHE: Dict[tuple, Optional[pd.DataFrame]] = {}


def _lineage_returns(methodology, protocol, prices) -> Optional[pd.DataFrame]:
    """Return series for every version of this concept the protocol can evaluate.

    Cached on the artifact hashes, so a page load does not re-run the whole
    lineage and a changed artifact invalidates naturally.
    """
    key = (methodology.concept, protocol.content_hash)
    if key in _LINEAGE_CACHE:
        return _LINEAGE_CACHE[key]

    series: Dict[str, pd.Series] = {}
    for version in MethodologyRegistry().versions(methodology.concept):
        if not assess_compatibility(version, protocol).compatible:
            continue
        try:
            other, _ = evaluate(version, protocol, prices)
            series[version.version_id] = other.daily_returns
        except Exception:
            continue

    frame = pd.DataFrame(series).dropna() if len(series) > 1 else None
    _LINEAGE_CACHE[key] = frame
    return frame


def _evaluate_full(methodology, protocol, prices, ledger, policies) -> Dict[str, Any]:
    """Run the three layers for one methodology, tolerating missing data."""
    out: Dict[str, Any] = {
        "result": None, "assessment": None, "policy_eval": None,
        "publication": None, "audit": None, "periods": None,
    }
    if prices is None or protocol is None:
        return out

    try:
        result, effective = evaluate(methodology, protocol, prices)
    except Exception:
        return out

    out["result"] = result
    out["audit"] = result.execution_audit
    out["periods"] = periods_per_year(protocol)

    trial_count = max(ledger.trial_count(methodology.concept, dsr_countable_only=True), 1)

    # PBO needs the comparable configurations from the same lineage. Omitting it
    # would leave the assessment PARTIAL for a reason that is a page-rendering
    # shortcut rather than a property of the methodology — which would misreport
    # the methodology.
    lineage = _lineage_returns(methodology, protocol, prices)

    factors = factor_model = None
    if protocol.benchmark and protocol.benchmark in prices.columns:
        bench = prices[protocol.benchmark].pct_change().dropna()
        factors = pd.DataFrame({"market": bench})
        factor_model = FactorModel(name="market-only", version=1, factors=("market",))

    try:
        assessment = assess(
            result.daily_returns, trial_count=trial_count,
            lineage_returns=lineage,
            factor_returns=factors, factor_model=factor_model,
        )
    except Exception:
        return out

    out["assessment"] = assessment

    policy = policies.resolve(DEFAULT_POLICY)
    evaluation = policy.evaluate(assessment, now=pd.Timestamp.now("UTC").isoformat())
    out["policy_eval"] = evaluation

    status = dict(result.result_status)
    status["statistical_assessment_complete"] = assessment.complete
    out["publication"] = decide(
        surface=Surface.PUBLIC_LIBRARY, result_status=status,
        assessment=assessment, policy_evaluation=evaluation,
    )
    return out


def _chain_state(version, protocol, bundle, calendars, ledger, graph, errata):
    """The artifact chain for one methodology version.

    Every page that shows a chain calls this. There was briefly a second
    implementation that built the same links as raw dicts for the detail page,
    which is precisely the drift `chain.py` warns about: the glyph and the table
    could disagree, and a disagreement between two renderings of the same fact is
    indistinguishable, to a reader, from the platform being wrong about the fact.
    """
    affecting = graph.findings_affecting(version.version_id)
    examined_here = [
        i for i in graph.investigations
        if any(ref.split("@")[0] == version.version_id.split("@")[0]
               for ref in i.examined)
    ]
    return build_chain_state(
        subject=version.version_id,
        findings=affecting,
        invalidating=[
            f for f in affecting
            if any(i.target == version.version_id
                   and i.relation.value == "INVALIDATES_RESULTS_OF"
                   for i in f.impacts)
        ],
        inquiries=[i for i in examined_here if i.is_open],
        null_results=[i for i in examined_here if i.produced_nothing],
        claims=graph.claims_for_methodology(version),
        assumptions=graph.assumptions_for_methodology(version),
        methodology=version,
        protocol=protocol,
        calendar_id=(
            calendars.resolve(protocol.walk_forward.calendar).calendar_id
            if protocol else None
        ),
        result=bundle.get("result"),
        assessment=bundle.get("assessment"),
        policy_evaluation=bundle.get("policy_eval"),
        publication=bundle.get("publication"),
        errata=errata,
        incomparable=[
            v for v in MethodologyRegistry().versions(version.concept)
            if v.version != version.version
            and version.contract.breaks_compatibility_with(v.contract)
        ],
    )


@router.get("/", response_class=HTMLResponse)
def library(request: Request):
    methodologies, protocols, policies, calendars, ledger = _registries()
    prices = _prices()
    errata = ledger.list_errata()
    graph = _graph()

    entries = []
    matrix_rows = []
    for concept, versions in methodologies.concepts().items():
        latest = methodologies.get(concept)
        protocol = _pick_protocol(latest, protocols)
        bundle = _evaluate_full(latest, protocol, prices, ledger, policies)
        breakdown = ledger.trial_breakdown(concept)
        publication = bundle.get("publication")

        chain = _chain_state(
            latest, protocol, bundle, calendars, ledger, graph, errata)
        matrix_rows.append({"version": latest, "chain": chain})

        entries.append({
            "chain": chain,
            "concept": concept,
            "title": latest.title,
            "objective": latest.objective,
            "latest_id": latest.version_id,
            "versions": versions,
            "attempted": breakdown["attempted_trials"],
            "dsr_countable": breakdown["dsr_countable_trials"],
            "citations": [c.identifier for c in latest.grounded_in],
            "errata": len(errata),
            "decision": publication.decision.value if publication else None,
            "validated": publication.may_claim_validated if publication else False,
        })

    # Every version of every concept, so the matrix answers "how does this
    # lineage stand?" without opening four pages.
    for concept in methodologies.concepts():
        for version in methodologies.versions(concept):
            if any(r["version"].version_id == version.version_id for r in matrix_rows):
                continue
            protocol = _pick_protocol(version, protocols)
            bundle = _evaluate_full(version, protocol, prices, ledger, policies)
            matrix_rows.append({
                "version": version,
                "chain": _chain_state(
                    version, protocol, bundle, calendars, ledger, graph, errata),
            })

    matrix_rows.sort(key=lambda r: (r["version"].concept, r["version"].version))

    return TEMPLATES.TemplateResponse(
        request, "index.html",
        {"entries": entries, "matrix": matrix_rows, "chain_order": CHAIN_ORDER},
    )


@router.get("/m/{concept}", response_class=HTMLResponse)
def concept_page(request: Request, concept: str):
    methodologies = MethodologyRegistry()
    try:
        latest = methodologies.get(concept)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return methodology_page(request, concept, latest.version)


@router.get("/m/{concept}/compare", response_class=HTMLResponse)
def compare_page(request: Request, concept: str, a: int, b: int):
    """Can these two versions' figures sit in the same table?

    The page arranges a prepared `ComparabilityView` and calculates nothing. In
    particular the eligibility of a performance visual is decided here, before
    the template runs, so a chart can never appear because a template forgot to
    check a condition.
    """
    methodologies, protocols, policies, calendars, ledger = _registries()
    try:
        left, right = methodologies.get(concept, a), methodologies.get(concept, b)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prices = _prices()
    right_bundle = _evaluate_full(right, _pick_protocol(right, protocols),
                                  prices, ledger, policies)
    publication = right_bundle.get("publication")

    # The performance class is a property of a recorded figure, not of a
    # methodology — so it is read from the ledger rows that would be charted.
    classes = {
        version.version_id: sorted({
            row["performance_class"] for row in ledger.list_performance(version.version_id)
        })
        for version in (left, right)
    }
    both_classified = all(classes[v.version_id] for v in (left, right))
    distinct = {c for cs in classes.values() for c in cs}

    view = build_comparability_view(
        left, right,
        publication_decision=publication.decision.value if publication else None,
        performance_class=(
            " · ".join(f"{vid.split('@')[-1]}: {', '.join(cs)}"
                       for vid, cs in classes.items())
            if both_classified else None
        ),
        # Two figures of different classes must not be linked into one series:
        # GIPS forbids joining actual to theoretical performance, and a reader
        # scanning a shared axis reads them as one track record.
        series_encoding_separated=len(distinct) <= 1,
    )

    return TEMPLATES.TemplateResponse(
        request, "compare.html",
        {
            "v": view,
            "a": left,
            "b": right,
            "t": version_timeline(concept, methodologies.versions(concept)),
        },
    )


@router.get("/m/{concept}/{version}", response_class=HTMLResponse)
def methodology_page(request: Request, concept: str, version: int):
    methodologies, protocols, policies, calendars, ledger = _registries()
    try:
        m = methodologies.get(concept, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    protocol = _pick_protocol(m, protocols)
    prices = _prices()
    bundle = _evaluate_full(m, protocol, prices, ledger, policies)
    graph = _graph()

    siblings = []
    for other in methodologies.versions(concept):
        siblings.append({
            "version": other.version,
            "change_rationale": other.change_rationale or "—",
            "comparable": not m.contract.breaks_compatibility_with(other.contract),
        })

    return TEMPLATES.TemplateResponse(
        request, "methodology.html",
        {
            "m": m,
            "protocol": protocol,
            "siblings": siblings,
            "errata": ledger.list_errata(),
            "t": version_timeline(
                concept, methodologies.versions(concept),
                errata_by_version={},
            ),
            "inquiries": [
                i for i in graph.investigations
                if any(ref.split("@")[0] == m.version_id.split("@")[0]
                       for ref in i.examined)
            ],
            "trial_reconciliation": {
                "ledger": ledger.trial_count(m.concept, dsr_countable_only=True),
                "investigations": graph.recorded_trials().get(m.version_id, 0),
            },
            "claims": graph.claims_for_methodology(m),
            "findings": graph.findings_affecting(m.version_id),
            "assumptions_declared": graph.assumptions_for_methodology(m),
            "chain": _chain_state(
                m, protocol, bundle, calendars, ledger, graph, ledger.list_errata()),
            **bundle,
        },
    )


@router.get("/protocols", response_class=HTMLResponse)
def protocols_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request, "protocols.html",
        {
            "protocols": ProtocolRegistry().load_all(),
            "calendars": CalendarRegistry().load_all(),
        },
    )


@router.get("/errata", response_class=HTMLResponse)
def errata_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request, "errata.html", {"errata": Ledger().list_errata()}
    )


@router.get("/claims", response_class=HTMLResponse)
def claims_page(request: Request):
    """Reasoning as artifacts: claims, the evidence bearing on them, and the
    assumptions everything rests on."""
    graph = _graph()
    assessments = [assess_claim(c, graph.evidence) for c in graph.claims]

    impacts = {
        a.claim.artifact_id: graph.impact_of_claim_change(a.claim.artifact_id)
        for a in assessments
    }
    return TEMPLATES.TemplateResponse(
        request, "claims.html",
        {
            "stance": {a.claim.artifact_id: claim_stance_projection(a) for a in assessments},
            "dependency": {
                a.artifact_id: assumption_dependency_projection(
                    a,
                    direct=[
                        m for m in graph.methodologies
                        if a.artifact_id in getattr(m, "assumptions_ref", ())
                    ],
                    inherited=[
                        m for m in graph.methodologies_depending_on_assumption(a.artifact_id)
                        if a.artifact_id not in getattr(m, "assumptions_ref", ())
                    ],
                )
                for a in graph.assumptions
            },
            "assessments": assessments,
            "impacts": impacts,
            "assumptions": graph.assumptions,
            "unvalidated": graph.unvalidated_assumptions(),
        },
    )


@router.get("/findings", response_class=HTMLResponse)
def findings_page(request: Request):
    """Investigations and their conclusions.

    A finding is neither a claim nor evidence: it is what someone concluded after
    synthesising several pieces of evidence, and it typically touches multiple
    claims, methodologies and assumptions at once.
    """
    graph = _graph()
    return TEMPLATES.TemplateResponse(
        request, "findings.html",
        {
            "projections": {
                f.artifact_id: finding_impact_projection(f, graph.evidence)
                for f in graph.findings
            },
            "provenance": [graph.finding_provenance(f) for f in graph.findings],
            "findings": graph.findings,
            "provisional": graph.provisional_findings(),
            "unevidenced": graph.unevidenced_findings(),
        },
    )


@router.get("/investigations", response_class=HTMLResponse)
def investigations_page(request: Request):
    """Questions asked — including the ones that produced nothing.

    A library that only publishes inquiries which found something reports a
    filtered history, and the filter runs in the direction that flatters the
    platform. Null and inconclusive results are given the same treatment as
    conclusive ones on purpose: the exit criterion for this work is that an
    inconclusive investigation is as presentable as a conclusive one.
    """
    graph = _graph()
    investigations = sorted(
        graph.investigations,
        key=lambda i: (i.opened_at or "", i.name),
        reverse=True,
    )
    return TEMPLATES.TemplateResponse(
        request, "investigations.html",
        {
            "provenance": [graph.investigation_provenance(i) for i in investigations],
            "investigations": investigations,
            "open": graph.open_inquiries(),
            "null_results": graph.null_results(),
            "unattributed": graph.unattributed_findings(),
            "trials_without_conclusion": sum(
                i.trials_examined for i in graph.null_results()
            ),
            "recorded_trials": graph.recorded_trials(),
        },
    )


@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    ledger = Ledger()
    return TEMPLATES.TemplateResponse(
        request, "runs.html", {"runs": ledger.list_runs()}
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str):
    """A run as a record, not a rendering.

    The page keeps two things apart that are easy to merge and misleading when
    merged: **what was recorded** when the run executed, and **what is true now**
    about the artifacts it used. A run evaluated under an older policy must keep
    showing the verdict it actually received; a claim refuted since must be shown
    as refuted *now*, not retro-fitted into the run's chain.
    """
    ledger = Ledger()
    run = ledger.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")

    concept, _, version = run["version_id"].removeprefix("methodology/").partition("@")

    # Current state of the artifacts this run used. Read now, labelled as now.
    graph = _graph()
    try:
        m = MethodologyRegistry().get(concept, int(version))
    except (KeyError, ValueError):
        m = None

    findings = graph.findings_affecting(run["version_id"])
    current = {
        "claims": graph.claims_for_methodology(m) if m else [],
        "assumptions": graph.assumptions_for_methodology(m) if m else [],
        "findings": findings,
        "invalidating": [
            f for f in findings
            if any(i.target == run["version_id"]
                   and i.relation.value == "INVALIDATES_RESULTS_OF"
                   for i in f.impacts)
        ],
        "errata": [e for e in ledger.list_errata()
                   if e.get("version_id") in (None, run["version_id"])],
        "is_latest": bool(m) and m.version == MethodologyRegistry().get(concept).version,
    }
    current["changed"] = bool(
        current["invalidating"] or current["errata"]
        or any(a.status.value in {"REFUTED", "CONTESTED", "SUPERSEDED"}
               for a in current["claims"])
    )

    # The recorded facts, judged by the standard in force now. Kept out of the
    # chain above on purpose: the run keeps the verdict it received.
    try:
        drift = evaluate_drift(
            run, PolicyRegistry().resolve(DEFAULT_POLICY),
            now=pd.Timestamp.now("UTC").isoformat(),
        )
    except KeyError:
        drift = None

    return TEMPLATES.TemplateResponse(
        request, "run.html",
        {"run": run, "concept": concept, "version": version,
         "chain": chain_from_record(run), "current": current, "drift": drift},
    )
