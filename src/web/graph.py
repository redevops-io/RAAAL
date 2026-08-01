"""Artifact graph and its view projections.

Two layers, deliberately separate:

* :class:`ArtifactGraph` — the canonical nodes and typed relationships. It hides
  nothing and orders nothing for display.
* :class:`GraphViewModel` — a *projection* answering one question with one
  layout. It may group, collapse or omit, but every such choice is recorded.

Keeping them apart stops the UI payload from becoming the canonical graph, which
would make display convenience indistinguishable from missing data.

**Nodes own state; edges own impact.** A finding is not globally adverse — it
`MOTIVATED` one methodology, `INVALIDATES_RESULTS_OF` another, `QUALIFIES` a
claim and `INTRODUCED` an assumption in the same breath. Putting adversity only
on the node would force one of those four truths onto all of them.

Every projection carries `omitted_count`. A view silently showing 12 of 40
dependents would be dishonest even where the visible 12 are correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from .chain import Adversity, State
from .semantics import resolve as resolve_relation


class Directness(str, Enum):
    DIRECT = "direct"
    INHERITED = "inherited"
    """Reached through another artifact — e.g. an assumption a methodology
    inherits via a claim it references rather than declaring itself."""

    HISTORICAL = "historical"
    """Held at some past point; retained, no longer current."""


class OmissionReason(str, Enum):
    """Why a projection dropped something.

    These are not interchangeable: a reader needs to know whether nodes are
    hidden for readability or because they lack access to them.
    """

    DISPLAY_LIMIT = "display_limit"
    PERMISSION_FILTER = "permission_filter"
    HISTORICAL_COLLAPSE = "historical_collapse"
    LOW_PRIORITY_GROUPING = "low_priority_grouping"


@dataclass(frozen=True)
class GraphNode:
    id: str
    artifact_type: str
    label: str
    state: State = State.UNKNOWN
    adversity: Adversity = Adversity.NONE
    """Derived summary for scanning. The relation-level truth stays on edges."""

    group: str = ""
    summary: str = ""
    href: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "artifact_type": self.artifact_type,
            "label": self.label,
            "state": self.state.value,
            "symbol": self.state.symbol,
            "adversity": self.adversity.value,
            "group": self.group,
            "summary": self.summary,
            "href": self.href,
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation_type: str
    detail: str = ""
    directness: Directness = Directness.DIRECT
    historical: bool = False

    @property
    def semantics(self):
        """Language and effect, from the one registry every component reads."""
        return resolve_relation(self.relation_type)

    @property
    def effect(self) -> Adversity:
        """What this relationship does to its target.

        Raises on an undeclared relation rather than defaulting to harmless — a
        silent default is how a genuinely adverse edge comes to render as benign.
        """
        return self.semantics.effect

    @property
    def semantic_key(self) -> tuple:
        """Canonical identity of a rendered relationship.

        Both the diagram and the fallback table expose this, so equivalence can
        be checked on *meaning* rather than on counts. Two renderings agreeing on
        how many edges exist while disagreeing about what one of them does would
        pass a count test and still be a lie.
        """
        return (
            self.source, self.relation_type, self.target,
            self.effect.value, self.directness.value,
        )

    def to_json(self) -> Dict[str, Any]:
        semantics = self.semantics
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "label": semantics.label,
            "effect": self.effect.value,
            "directness": self.directness.value,
            "historical": self.historical,
            "detail": self.detail,
            "semantic_key": "|".join(self.semantic_key),
        }


@dataclass
class ArtifactGraph:
    """Canonical nodes and typed relationships. Hides nothing."""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> GraphNode:
        self.nodes.setdefault(node.id, node)
        return self.nodes[node.id]

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def neighbourhood(self, root_ids: Sequence[str]) -> List[GraphEdge]:
        roots = set(root_ids)
        return [e for e in self.edges if e.source in roots or e.target in roots]


@dataclass
class GraphViewModel:
    """A projection of the artifact graph for one question and layout.

    `omitted_count` is not optional bookkeeping. Any projection that drops nodes
    must say how many, or the view claims completeness it does not have.
    """

    projection_type: str
    root_ids: Sequence[str]
    nodes: Sequence[GraphNode]
    edges: Sequence[GraphEdge]
    layout_hint: str = "impact"
    omitted_count: int = 0
    omission_reason: Optional[OmissionReason] = None
    grouping_rules: Sequence[str] = ()
    source_graph_version: str = "0.1"

    # ---- equivalence -------------------------------------------------------

    def fallback_rows(self) -> List[Dict[str, Any]]:
        """Textual rendering of every edge, from this same payload.

        The bidirectional equivalence test compares these rows against what the
        visual actually draws, in both directions, so neither rendering can
        quietly omit an awkward relationship.
        """
        by_id = {n.id: n for n in self.nodes}
        rows = []
        for edge in self.edges:
            source, target = by_id.get(edge.source), by_id.get(edge.target)
            rows.append({
                **edge.to_json(),
                "source_label": source.label if source else edge.source,
                "target_label": target.label if target else edge.target,
                "source_type": source.artifact_type if source else "",
                "target_type": target.artifact_type if target else "",
            })
        return rows

    def semantic_keys(self) -> Set[tuple]:
        """Every rendered relationship, by meaning. The equivalence anchor."""
        return {e.semantic_key for e in self.edges}

    def accessibility_summary(self) -> str:
        """One sentence a screen reader can use in place of the diagram."""
        if not self.edges:
            return f"{len(self.nodes)} artifacts, no relationships shown."
        counts: Dict[str, int] = {}
        for edge in self.edges:
            counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
        described = ", ".join(
            f"{n} {rel.replace('_', ' ').lower()}" for rel, n in sorted(counts.items())
        )
        tail = (
            f"; {self.omitted_count} omitted "
            f"({(self.omission_reason or OmissionReason.DISPLAY_LIMIT).value.replace('_', ' ')})"
            if self.omitted_count else ""
        )
        return f"{len(self.nodes)} artifacts linked by {described}{tail}."

    def compact_summary(self) -> str:
        """Plain-language one-liner — the middle tier of progressive disclosure."""
        blocking = [e for e in self.edges if e.effect is Adversity.BLOCKING]
        advisory = [e for e in self.edges if e.effect is Adversity.ADVISORY]
        if blocking:
            return f"{len(blocking)} adverse relationship(s), {len(self.edges)} total"
        if advisory:
            return f"{len(advisory)} qualifying relationship(s), {len(self.edges)} total"
        return f"{len(self.edges)} relationship(s), none adverse"

    # ---- layout ------------------------------------------------------------

    #: Groups that sit to the left of the root. Everything else sits right.
    _INPUT_GROUPS = frozenset({"input", "supports", "qualifies", "contradicts",
                               "dependent"})

    def layout(self) -> Dict[str, Any]:
        """Positioned nodes and edges, computed here rather than in a template.

        Templates should draw, not calculate. Keeping the geometry in the payload
        means the diagram and the table are still one source of truth, and a
        layout bug cannot silently drop an edge the table still lists.
        """
        roots = set(self.root_ids)
        inputs = [n for n in self.nodes if n.id not in roots and n.group in self._INPUT_GROUPS]
        outputs = [n for n in self.nodes if n.id not in roots and n.group not in self._INPUT_GROUPS]

        row = 40
        gap = 34
        height = max(len(inputs), len(outputs), 1) * gap + 60
        mid = height / 2

        pos: Dict[str, tuple] = {}
        for i, node in enumerate(inputs):
            pos[node.id] = (14, row + i * gap)
        for i, node in enumerate(outputs):
            pos[node.id] = (470, row + i * gap)
        for node_id in roots:
            pos[node_id] = (250, mid)

        placed_nodes = [
            {**n.to_json(), "x": pos[n.id][0], "y": pos[n.id][1],
             "anchor": "start" if n.id not in roots or True else "middle"}
            for n in self.nodes if n.id in pos
        ]

        placed_edges = []
        for edge in self.edges:
            if edge.source not in pos or edge.target not in pos:
                continue
            sx, sy = pos[edge.source]
            tx, ty = pos[edge.target]
            # Start past the source label, stop before the target label.
            x1, x2 = (sx + 200, tx - 6) if sx < tx else (sx - 6, tx + 200)
            placed_edges.append({
                **edge.to_json(),
                "x1": x1, "y1": sy - 4, "x2": x2, "y2": ty - 4,
                "lx": (x1 + x2) / 2, "ly": (sy + ty) / 2 - 8,
            })

        return {"width": 760, "height": height,
                "nodes": placed_nodes, "edges": placed_edges}

    def to_json(self) -> Dict[str, Any]:
        return {
            "projection_type": self.projection_type,
            "root_ids": list(self.root_ids),
            "layout_hint": self.layout_hint,
            "omitted_count": self.omitted_count,
            "omission_reason": self.omission_reason.value if self.omission_reason else None,
            "grouping_rules": list(self.grouping_rules),
            "source_graph_version": self.source_graph_version,
            "nodes": [n.to_json() for n in self.nodes],
            "edges": [e.to_json() for e in self.edges],
            "fallback_rows": self.fallback_rows(),
            "accessibility_summary": self.accessibility_summary(),
            "compact_summary": self.compact_summary(),
        }


# --- projections -----------------------------------------------------------


def _claim_state(status: str) -> State:
    return {
        "SUPPORTED": State.OK, "QUALIFIED": State.WARN, "CONTESTED": State.WARN,
        "REFUTED": State.BLOCK, "SUPERSEDED": State.BLOCK,
        "UNASSESSED": State.UNKNOWN,
    }.get(status, State.UNKNOWN)


def finding_impact_projection(finding, evidence: Sequence[Any]) -> GraphViewModel:
    """Many inputs → one synthesis → many typed impacts.

    The layout that makes a finding legible: evidence flows in from the left, the
    conclusion sits at centre, impacts fan right with their relation on the edge.
    """
    graph = ArtifactGraph()
    root = f"finding/{finding.name}@{finding.version}"

    graph.add_node(GraphNode(
        id=root, artifact_type="finding", label=root, group="synthesis",
        state=State.OK if finding.is_evidenced else State.WARN,
        summary=finding.statement[:160],
        href="/ui/findings",
    ))

    cited = {e.artifact_id for e in evidence} & set(finding.supported_by)
    for item in evidence:
        if item.artifact_id not in cited:
            continue
        graph.add_node(GraphNode(
            id=item.artifact_id, artifact_type="evidence",
            label=item.identifier or item.artifact_id, group="input",
            state={"SUPPORTS": State.OK, "CONTRADICTS": State.BLOCK}.get(
                item.stance.value, State.WARN),
            summary=item.summary[:140],
        ))
        graph.add_edge(GraphEdge(
            source=item.artifact_id, target=root,
            relation_type=item.stance.value, detail=item.strength,
        ))

    for impact in finding.impacts:
        kind = impact.target.split("/", 1)[0]
        graph.add_node(GraphNode(
            id=impact.target, artifact_type=kind, label=impact.target, group="impact",
            state=State.UNKNOWN,
        ))
        graph.add_edge(GraphEdge(
            source=root, target=impact.target,
            relation_type=impact.relation.value, detail=impact.detail,
        ))

    return GraphViewModel(
        projection_type="finding_impact",
        root_ids=(root,),
        nodes=tuple(graph.nodes.values()),
        edges=tuple(graph.edges),
        layout_hint="impact",
        grouping_rules=("input", "synthesis", "impact"),
    )


def assumption_dependency_projection(
    assumption, direct: Sequence[Any], inherited: Sequence[Any], limit: int = 24
) -> GraphViewModel:
    """One premise → direct and inherited dependents.

    Directness is the fact that matters most here, and it gets its own visual
    channel (line style) rather than being folded into adversity.
    """
    graph = ArtifactGraph()
    root = assumption.artifact_id

    graph.add_node(GraphNode(
        id=root, artifact_type="assumption", label=root, group="premise",
        state=State.OK if assumption.is_validated else State.WARN,
        adversity=Adversity.NONE if assumption.is_validated else Adversity.ADVISORY,
        summary=assumption.statement[:160],
        href="/ui/claims#assumptions",
    ))

    omitted = 0
    seen: Set[str] = set()
    for group, directness in ((direct, Directness.DIRECT),
                              (inherited, Directness.INHERITED)):
        for m in group:
            if m.version_id in seen:
                continue
            if len(seen) >= limit:
                omitted += 1
                continue
            seen.add(m.version_id)
            concept, _, version = m.version_id.removeprefix("methodology/").partition("@")
            graph.add_node(GraphNode(
                id=m.version_id, artifact_type="methodology", label=m.version_id,
                group="dependent", state=State.UNKNOWN,
                href=f"/ui/m/{concept}/{version}",
            ))
            graph.add_edge(GraphEdge(
                source=m.version_id, target=root, relation_type="DEPENDS_ON",
                directness=directness,
                detail=("declared" if directness is Directness.DIRECT
                        else "inherited via a referenced claim"),
            ))

    return GraphViewModel(
        projection_type="assumption_dependency",
        root_ids=(root,),
        nodes=tuple(graph.nodes.values()),
        edges=tuple(graph.edges),
        layout_hint="dependency",
        omitted_count=omitted,
        omission_reason=OmissionReason.DISPLAY_LIMIT if omitted else None,
        grouping_rules=("premise", "dependent"),
    )


def claim_stance_projection(assessment) -> GraphViewModel:
    """Multiple evidence stances → one derived claim state.

    Adversity is deliberately not the visible dimension here. Stance and strength
    are, because turning every relation into a red/amber/green scale would lose
    the distinction between refuting a claim and narrowing where it holds.
    """
    graph = ArtifactGraph()
    claim = assessment.claim
    root = claim.artifact_id

    graph.add_node(GraphNode(
        id=root, artifact_type="claim", label=root, group="claim",
        state=_claim_state(assessment.status.value),
        summary=claim.statement[:160],
        href=f"/ui/claims#{claim.name}",
    ))

    for stance, items in (
        ("SUPPORTS", assessment.supporting),
        ("QUALIFIES", assessment.qualifying),
        ("CONTRADICTS", assessment.contradicting),
    ):
        for item in items:
            graph.add_node(GraphNode(
                id=item.artifact_id, artifact_type="evidence",
                label=item.identifier or item.artifact_id, group=stance.lower(),
                state={"SUPPORTS": State.OK, "CONTRADICTS": State.BLOCK}.get(
                    stance, State.WARN),
                summary=item.summary[:140],
            ))
            graph.add_edge(GraphEdge(
                source=item.artifact_id, target=root,
                relation_type=stance, detail=item.strength,
            ))

    return GraphViewModel(
        projection_type="claim_stance",
        root_ids=(root,),
        nodes=tuple(graph.nodes.values()),
        edges=tuple(graph.edges),
        layout_hint="stance",
        grouping_rules=("supports", "qualifies", "contradicts"),
    )


# --- version timeline ------------------------------------------------------


@dataclass(frozen=True)
class TimelineSegment:
    """The span between two adjacent versions of one lineage.

    `comparable` and `blockers` come from the same comparability engine the
    compare page uses. A template must never author a wall: a visual line may
    cross a version boundary only where that engine says the boundary is
    traversable.
    """

    from_version: str
    to_version: str
    comparable: bool
    blockers: Sequence[str] = ()
    change_rationale: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "comparable": self.comparable,
            "blockers": list(self.blockers),
            "change_rationale": self.change_rationale,
        }


@dataclass
class TimelineViewModel:
    """A lineage as a track, with comparability boundaries made spatial."""

    concept: str
    versions: Sequence[Dict[str, Any]]
    segments: Sequence[TimelineSegment]

    @property
    def walls(self) -> List[TimelineSegment]:
        return [s for s in self.segments if not s.comparable]

    def compact_summary(self) -> str:
        walls = len(self.walls)
        if not walls:
            return f"{len(self.versions)} versions, all comparable"
        return (
            f"{len(self.versions)} versions, {walls} comparability "
            f"{'wall' if walls == 1 else 'walls'}"
        )

    def accessibility_summary(self) -> str:
        if not self.walls:
            return f"{self.concept}: {len(self.versions)} versions, no comparability breaks."
        described = "; ".join(
            f"between {w.from_version} and {w.to_version}: {', '.join(w.blockers)}"
            for w in self.walls
        )
        return f"{self.concept}: comparability breaks {described}."

    def layout(self) -> Dict[str, Any]:
        """Positions computed here, not in a template."""
        step = 170
        stops = [
            {**v, "x": 30 + i * step, "y": 46}
            for i, v in enumerate(self.versions)
        ]
        by_id = {v["version_id"]: v for v in stops}
        spans = []
        for segment in self.segments:
            a, b = by_id.get(segment.from_version), by_id.get(segment.to_version)
            if not a or not b:
                continue
            spans.append({
                **segment.to_json(),
                "x1": a["x"] + 96, "x2": b["x"] - 6, "y": 46,
                "mid": (a["x"] + 96 + b["x"] - 6) / 2,
            })
        return {"width": max(360, 30 + len(stops) * step), "height": 110,
                "stops": stops, "spans": spans}

    def to_json(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "versions": list(self.versions),
            "segments": [s.to_json() for s in self.segments],
            "walls": len(self.walls),
            "compact_summary": self.compact_summary(),
            "accessibility_summary": self.accessibility_summary(),
        }


def version_timeline(concept: str, versions: Sequence[Any],
                     errata_by_version: Optional[Dict[str, int]] = None) -> TimelineViewModel:
    """Build a lineage timeline. Walls are derived, never declared."""
    errata_by_version = errata_by_version or {}
    ordered = sorted(versions, key=lambda m: m.version)

    stops = [
        {
            "version_id": m.version_id,
            "version": m.version,
            "concept": m.concept,
            "deprecated": bool(m.deprecation_date),
            "errata": errata_by_version.get(m.version_id, 0),
            "latest": m.version == ordered[-1].version,
        }
        for m in ordered
    ]

    segments = []
    for older, newer in zip(ordered, ordered[1:]):
        breaks = newer.contract.breaks_compatibility_with(older.contract)
        segments.append(TimelineSegment(
            from_version=older.version_id,
            to_version=newer.version_id,
            comparable=not breaks,
            blockers=tuple(breaks),
            change_rationale=newer.change_rationale,
        ))

    return TimelineViewModel(concept=concept, versions=stops, segments=tuple(segments))
