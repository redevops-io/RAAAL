"""Chain state — one payload, two renderings.

The artifact chain is the product's visual identity, and this module is the
single place its state is computed. Both the glyph and the textual table are
generated from the payload this returns; neither may infer or recompute a status.

That is a testable invariant rather than a style preference: if the visual and
the text can disagree, the visual has become a second source of truth, and by the
project's own principle (`artifacts own declarations, relationships own
derivations`) a second source always drifts.

The chain spans three semantic domains, and the glyph groups them so it does not
read as a build pipeline where every dot should turn green::

    reasoning   execution        judgment
    ● ● ●   │   ● ● ● ●   │   ● ✕ ●

**A dot reports the state of an artifact class, not its presence.** Otherwise a
methodology looks healthy because many artifacts exist, when the salient fact is
that one of them establishes a blocker.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class State(str, Enum):
    """Chain-link state. Symbols carry meaning without colour."""

    OK = "ok"
    WARN = "warn"
    BLOCK = "block"
    ABSENT = "absent"
    UNKNOWN = "unknown"

    @property
    def symbol(self) -> str:
        return {
            State.OK: "●",
            State.WARN: "◐",
            State.BLOCK: "✕",
            State.ABSENT: "○",
            State.UNKNOWN: "·",
        }[self]

    @property
    def label(self) -> str:
        return {
            State.OK: "clear",
            State.WARN: "attention",
            State.BLOCK: "blocked",
            State.ABSENT: "none",
            State.UNKNOWN: "not assessed",
        }[self]


class Domain(str, Enum):
    REASONING = "reasoning"
    EXECUTION = "execution"
    JUDGMENT = "judgment"


class Adversity(str, Enum):
    """*How* a link is adverse — not merely *that* it is.

    Collapsing these was a real defect: "blocked at errata" asserts errata are
    failures, when an erratum can be present without currently blocking
    anything. A finding that MOTIVATED a methodology is likewise adverse in no
    sense at all.
    """

    NONE = "none"
    ADVISORY = "advisory"
    """Worth knowing; changes nothing about publication."""

    AFFECTING = "affecting"
    """An adverse historical condition attached to the lineage. Not a blocker."""

    BLOCKING = "blocking"
    """Actively prevents publication."""


@dataclass(frozen=True)
class Link:
    """One position in the chain. Position is stable across every row.

    `summary` is the plain-language one-liner of progressive disclosure; `detail`
    is what expands. Both are text, so the table rendering needs no extra work.
    """

    step: str
    domain: Domain
    state: State
    value: str
    summary: str = ""
    detail: str = ""
    href: Optional[str] = None
    adversity: Adversity = Adversity.NONE

    @property
    def aria(self) -> str:
        """Accessible label. Also the string the equivalence test compares."""
        return f"{self.step}: {self.state.label} — {self.value}"

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "domain": self.domain.value,
            "state": self.state.value,
            "symbol": self.state.symbol,
            "value": self.value,
            "summary": self.summary,
            "detail": self.detail,
            "href": self.href,
            "adversity": self.adversity.value,
            "aria": self.aria,
        }


#: Canonical order. Position must be stable across every rendering, because
#: column-scanning a table only works if a given step is always in the same slot.
CHAIN_ORDER: Sequence[tuple[str, Domain]] = (
    ("Inquiry", Domain.REASONING),
    ("Findings", Domain.REASONING),
    ("Claims", Domain.REASONING),
    ("Assumptions", Domain.REASONING),
    ("Methodology", Domain.EXECUTION),
    ("Protocol", Domain.EXECUTION),
    ("Calendar", Domain.EXECUTION),
    ("Run", Domain.EXECUTION),
    ("Assessment", Domain.JUDGMENT),
    ("Policy", Domain.JUDGMENT),
    ("Publication", Domain.JUDGMENT),
    ("Errata", Domain.JUDGMENT),
    ("Comparability", Domain.JUDGMENT),
)


@dataclass(frozen=True)
class ChainState:
    """The payload. Everything rendered anywhere comes from here."""

    subject: str
    links: Sequence[Link]

    def by_domain(self) -> List[tuple[Domain, List[Link]]]:
        """Links grouped for the glyph, preserving canonical order."""
        out: List[tuple[Domain, List[Link]]] = []
        for domain in Domain:
            members = [l for l in self.links if l.domain is domain]
            if members:
                out.append((domain, members))
        return out

    @property
    def worst(self) -> State:
        """The state a scanning reader should take away.

        Deliberately pessimistic: one blocker outranks ten clear links, because
        the salient fact about a blocked methodology is that it is blocked.
        """
        for state in (State.BLOCK, State.WARN, State.UNKNOWN, State.ABSENT):
            if any(l.state is state for l in self.links):
                return state
        return State.OK

    @property
    def headline(self) -> str:
        """One line, plain language, preserving the kind of adversity.

        A summary reading "blocked at findings, errata" flattened three different
        meanings into one. The severity grammar is fixed across every row — what
        stops use, then what warrants review, then what forms part of the
        historical record — so a scanning reader learns the shape once.
        """
        blocking = [l for l in self.links if l.adversity is Adversity.BLOCKING]
        advisory = [l for l in self.links if l.adversity is Adversity.ADVISORY]
        affecting = [l for l in self.links if l.adversity is Adversity.AFFECTING]

        def _phrase(link: "Link") -> str:
            count = link.value.split()[0]
            noun = link.step.lower()
            if count == "1":
                # English is not regular here — "errata" is already plural.
                noun = _SINGULAR.get(noun, noun[:-1] if noun.endswith("s") else noun)
            return f"{count} {noun}"

        parts: List[str] = []
        if blocking:
            parts.append("Blocked at " + ", ".join(l.step.lower() for l in blocking))
        if advisory and not blocking:
            parts.append("attention at " + ", ".join(l.step.lower() for l in advisory))
        if affecting:
            parts.append("affected by " + ", ".join(_phrase(l) for l in affecting))

        return " · ".join(parts) if parts else "Clear across the chain"

    def glyph_text(self) -> str:
        """Textual glyph, grouped. The equivalence anchor for the SVG/HTML row."""
        return "  │  ".join(
            " ".join(link.state.symbol for link in members)
            for _, members in self.by_domain()
        )

    def as_table(self) -> List[Dict[str, Any]]:
        """Textual rendering. Same payload, no recomputation."""
        return [link.to_json() for link in self.links]

    def to_json(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "worst": self.worst.value,
            "headline": self.headline,
            "glyph": self.glyph_text(),
            "links": self.as_table(),
        }


#: Irregular plurals in chain step names. Kept explicit rather than inferred,
#: because a wrong singular reads as carelessness in a product about precision.
_SINGULAR = {"errata": "erratum", "findings": "finding", "claims": "claim",
             "assumptions": "assumption"}


def _link(step: str, state: State, value: str, **kw) -> Link:
    domain = next(d for s, d in CHAIN_ORDER if s == step)
    return Link(step=step, domain=domain, state=state, value=value, **kw)


def chain_from_record(run: Dict[str, Any]) -> ChainState:
    """Rebuild a run's chain from what was persisted at execution time.

    The run page must never recompute a verdict. A run evaluated under an older
    policy has to keep showing the verdict it actually received, otherwise the
    ledger stops being a record and becomes a rendering of today's opinion.

    The reasoning links are deliberately `UNKNOWN` rather than filled from the
    current knowledge graph: claims and findings move after a run, and quietly
    inserting today's values into a historical chain is exactly the merge this
    page exists to prevent. Current reasoning appears in its own panel, labelled.
    """
    version_id = run["version_id"]
    concept, _, version = version_id.removeprefix("methodology/").partition("@")

    not_recorded = "not recorded at execution time"
    links: List[Link] = [
        _link("Inquiry", State.UNKNOWN, "—", summary=not_recorded),
        _link("Findings", State.UNKNOWN, "—", summary=not_recorded),
        _link("Claims", State.UNKNOWN, "—", summary=not_recorded),
        _link("Assumptions", State.UNKNOWN, "—", summary=not_recorded),
        _link("Methodology", State.OK, version_id,
              summary=f"hash {run.get('methodology_hash', '')[:12]}…"
                      if run.get("methodology_hash") else "as executed",
              href=f"/ui/m/{concept}/{version}"),
        _link("Protocol", State.OK, run["protocol_id"],
              summary="as executed", href="/ui/protocols"),
        _link("Calendar", State.OK if run.get("calendar_id") else State.UNKNOWN,
              run.get("calendar_id") or "—",
              summary="sessions the result was measured over",
              href="/ui/protocols" if run.get("calendar_id") else None),
        _link("Run", State.OK, run["run_id"],
              summary=f"trial ordinal {run['trial_ordinal']} · {run['outcome']}"),
    ]

    assessment = run.get("assessment") or {}
    if assessment:
        status = assessment.get("computation_status", "—")
        links.append(_link(
            "Assessment", State.OK if status == "VALID" else State.WARN, status,
            summary=f"{assessment.get('trial_count')} configuration(s) counted",
        ))
    else:
        links.append(_link("Assessment", State.UNKNOWN, "—", summary=not_recorded))

    policy = run.get("policy_evaluation") or {}
    if policy:
        status = policy.get("status")
        links.append(_link(
            "Policy",
            {"PASS": State.OK, "WARN": State.WARN,
             "FAIL": State.BLOCK}.get(status, State.UNKNOWN),
            policy.get("policy_id", "—"),
            summary=f"{status} · evidence {policy.get('evidence_grade')}",
            adversity={"PASS": Adversity.NONE, "WARN": Adversity.ADVISORY,
                       "FAIL": Adversity.BLOCKING}.get(status, Adversity.NONE),
        ))
    else:
        links.append(_link("Policy", State.UNKNOWN, "—", summary=not_recorded))

    publication = run.get("publication_decision") or {}
    if publication:
        decision = publication.get("decision")
        links.append(_link(
            "Publication",
            {"ALLOW": State.OK, "ALLOW_WITH_DISCLOSURE": State.WARN,
             "BLOCK": State.BLOCK}.get(decision, State.UNKNOWN),
            decision or "—",
            summary=("may claim validated" if publication.get("may_claim_validated")
                     else "not validated"),
            adversity={"ALLOW": Adversity.NONE,
                       "ALLOW_WITH_DISCLOSURE": Adversity.ADVISORY,
                       "BLOCK": Adversity.BLOCKING}.get(decision, Adversity.NONE),
        ))
    else:
        links.append(_link("Publication", State.UNKNOWN, "—", summary=not_recorded))

    links.append(_link("Errata", State.UNKNOWN, "—",
                       summary="corrections are a fact about the lineage now, "
                               "not about this run then"))
    links.append(_link("Comparability", State.UNKNOWN, "—",
                       summary="comparability is a property of the lineage today"))

    return ChainState(subject=run["run_id"], links=tuple(links))


def build_chain_state(
    *,
    subject: str,
    findings: Sequence[Any] = (),
    invalidating: Sequence[Any] = (),
    claims: Sequence[Any] = (),
    assumptions: Sequence[Any] = (),
    inquiries: Sequence[Any] = (),
    null_results: Sequence[Any] = (),
    methodology=None,
    protocol=None,
    calendar_id: Optional[str] = None,
    result=None,
    assessment=None,
    policy_evaluation=None,
    publication=None,
    errata: Sequence[Any] = (),
    incomparable: Sequence[Any] = (),
) -> ChainState:
    """Compute chain state once, for every rendering.

    Each link's state reflects the *condition* of that artifact class — a claim
    link is BLOCK when a referenced claim is refuted, not merely OK because
    claims exist.
    """
    links: List[Link] = []

    # Open inquiries are not adverse — an unanswered question about a
    # methodology is a normal state, and marking it as a defect would create an
    # incentive to close questions rather than to ask them. Null results are
    # likewise not failures; they are spent trials that produced no conclusion,
    # and the only harmful thing about them is going unrecorded.
    spent = sum(getattr(i, "trials_examined", 0) for i in null_results)
    links.append(_link(
        "Inquiry",
        State.WARN if inquiries else State.OK,
        f"{len(inquiries)} open",
        summary=(
            (f"{len(inquiries)} question(s) still open"
             if inquiries else "no open questions")
            + (f" · {spent} trial(s) spent with no conclusion" if spent else "")
        ),
        adversity=Adversity.ADVISORY if inquiries else Adversity.NONE,
        href="/ui/investigations",
    ))

    unhealthy = [
        a for a in claims
        if getattr(a, "status", None) is not None
        and a.status.value in {"REFUTED", "CONTESTED", "SUPERSEDED"}
    ]
    unvalidated = [a for a in assumptions if not getattr(a, "is_validated", True)]

    # A finding that MOTIVATED this version is not adverse at all; only one that
    # invalidates its results is. Presence alone must not read as failure.
    links.append(_link(
        "Findings",
        State.BLOCK if invalidating else (State.WARN if findings else State.OK),
        f"{len(findings)} concluded",
        summary=(
            f"{len(invalidating)} invalidate results of this version"
            if invalidating else
            ("investigations touch this version" if findings else "no findings")
        ),
        adversity=(
            Adversity.AFFECTING if invalidating
            else (Adversity.ADVISORY if findings else Adversity.NONE)
        ),
        href="/ui/findings",
    ))

    links.append(_link(
        "Claims",
        State.BLOCK if unhealthy else (State.OK if claims else State.ABSENT),
        f"{len(claims)} referenced",
        summary=(
            ", ".join(f"{a.claim.name}: {a.status.value}" for a in claims)
            or "none referenced"
        ),
        adversity=Adversity.BLOCKING if unhealthy else Adversity.NONE,
        href="/ui/claims",
    ))

    links.append(_link(
        "Assumptions",
        State.WARN if unvalidated else (State.OK if assumptions else State.ABSENT),
        f"{len(assumptions)} declared",
        summary=(
            f"{len(unvalidated)} unvalidated"
            if unvalidated else "all have a checking test"
        ),
        adversity=Adversity.ADVISORY if unvalidated else Adversity.NONE,
        href="/ui/claims#assumptions",
    ))

    links.append(_link(
        "Methodology",
        State.OK if methodology else State.UNKNOWN,
        methodology.version_id if methodology else "—",
        summary=f"hash {methodology.content_hash[:12]}…" if methodology else "",
    ))

    if protocol is None:
        links.append(_link("Protocol", State.BLOCK, "none compatible",
                           summary="no published protocol can validly evaluate this"))
        links.append(_link("Calendar", State.UNKNOWN, "—"))
    else:
        links.append(_link(
            "Protocol", State.OK, protocol.protocol_id,
            summary=(
                f"{protocol.transaction_costs.bps}bps · "
                f"lag {protocol.transaction_costs.execution_lag_days}d · "
                f"warmup {protocol.walk_forward.warmup}"
            ),
            href="/ui/protocols",
        ))
        links.append(_link(
            "Calendar",
            State.OK if calendar_id else State.BLOCK,
            calendar_id or protocol.walk_forward.calendar,
            summary="sessions the result was measured over",
            href="/ui/protocols",
        ))

    if result is None:
        links.append(_link("Run", State.WARN, "not evaluated",
                           summary="no price snapshot available"))
    else:
        links.append(_link(
            "Run", State.OK, f"{result.n_rebalances} rebalances",
            summary=f"{result.n_observations} sessions · "
                    f"{result.period_start} … {result.period_end}",
        ))

    if assessment is None:
        links.append(_link("Assessment", State.UNKNOWN, "—"))
    else:
        links.append(_link(
            "Assessment",
            State.OK if assessment.computation_status == "VALID" else State.WARN,
            assessment.computation_status,
            summary=f"{assessment.trial_count} configuration(s) counted",
        ))

    if policy_evaluation is None:
        links.append(_link("Policy", State.UNKNOWN, "—"))
    else:
        links.append(_link(
            "Policy",
            {"PASS": State.OK, "WARN": State.WARN,
             "FAIL": State.BLOCK}[policy_evaluation.status.value],
            policy_evaluation.policy_id,
            summary=f"{policy_evaluation.status.value} · "
                    f"evidence {policy_evaluation.evidence_grade.value}",
            adversity={
                "PASS": Adversity.NONE, "WARN": Adversity.ADVISORY,
                "FAIL": Adversity.BLOCKING,
            }[policy_evaluation.status.value],
        ))

    if publication is None:
        links.append(_link("Publication", State.UNKNOWN, "—"))
    else:
        links.append(_link(
            "Publication",
            {"ALLOW": State.OK, "ALLOW_WITH_DISCLOSURE": State.WARN,
             "BLOCK": State.BLOCK}[publication.decision.value],
            publication.decision.value,
            summary=("may claim validated" if publication.may_claim_validated
                     else "not validated"),
            adversity={
                "ALLOW": Adversity.NONE,
                "ALLOW_WITH_DISCLOSURE": Adversity.ADVISORY,
                "BLOCK": Adversity.BLOCKING,
            }[publication.decision.value],
        ))

    # An erratum is an adverse historical condition, not an active blocker. The
    # publication gate decides what may be shown; a correction having been
    # published is a fact about the lineage's history.
    links.append(_link(
        "Errata",
        State.WARN if errata else State.OK,
        f"{len(errata)} affecting lineage",
        summary="corrections published against this lineage" if errata else "none",
        adversity=Adversity.AFFECTING if errata else Adversity.NONE,
        href="/ui/errata" if errata else None,
    ))

    # Incomparability is not a defect. It is the contract working: a version that
    # cannot be read against its siblings is one whose promise genuinely changed.
    # Rendering it as a failure would teach readers to avoid the very versioning
    # discipline the platform exists to enforce.
    links.append(_link(
        "Comparability",
        State.WARN if incomparable else State.OK,
        (f"not comparable with {len(incomparable)} version(s)"
         if incomparable else "comparable across lineage"),
        summary=(
            ", ".join(getattr(v, "version_id", str(v)) for v in incomparable)
            if incomparable else "figures can be read against every sibling"
        ),
        adversity=Adversity.ADVISORY if incomparable else Adversity.NONE,
    ))

    return ChainState(subject=subject, links=tuple(links))
